"""
Azure AI Search Multimodal Embedding Pipeline

This script implements a multimodal embedding pipeline using Azure AI Search's
indexer and skillset architecture. It processes PDFs from Azure Blob Storage,
extracts text and images, generates image descriptions using GenAI, and creates
searchable embeddings for both text and images.

Usage:
    python azure_search_pipeline.py --create-all
    python azure_search_pipeline.py --run-indexer
    python azure_search_pipeline.py --query "energy" --top 5
    python azure_search_pipeline.py --status
    python azure_search_pipeline.py --delete-all
"""

import os
import sys
import time
import argparse
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        ComplexField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
        AzureOpenAIVectorizer,
        AzureOpenAIParameters,
        SemanticConfiguration,
        SemanticPrioritizedFields,
        SemanticField,
        SemanticSearch,
        SearchIndexerDataSourceConnection,
        SearchIndexerDataContainer,
        SearchIndexer,
        SearchIndexerSkillset,
        InputFieldMappingEntry,
        OutputFieldMappingEntry,
        FieldMapping,
        DocumentExtractionSkill,
        SplitSkill,
        AzureOpenAIEmbeddingSkill,
        WebApiSkill,
        ShaperSkill,
        IndexProjectionMode,
        SearchIndexerIndexProjections,
        SearchIndexerIndexProjectionSelector,
        SearchIndexerIndexProjectionsParameters,
        SearchIndexerKnowledgeStore,
        SearchIndexerKnowledgeStoreProjection,
        SearchIndexerKnowledgeStoreFileProjectionSelector,
    )
except ImportError:
    print("Error: Required Azure SDK packages not installed.")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


class AzureSearchPipeline:
    """Main class for managing Azure AI Search multimodal embedding pipeline."""
    
    def __init__(self):
        """Initialize the pipeline with configuration from environment variables."""
        load_dotenv()
        
        # Validate required environment variables
        self._validate_env()
        
        # Azure AI Search configuration
        self.search_endpoint = os.getenv("SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        
        # Azure Storage configuration
        self.storage_connection = os.getenv("STORAGE_CONNECTION_STRING")
        self.pdf_container = os.getenv("PDF_CONTAINER", "sustainable-ai-pdf")
        self.image_container = os.getenv("IMAGE_CONTAINER", "sustainable-ai-pdf-images")
        
        # Azure OpenAI configuration
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        
        # Chat completion configuration
        self.chat_completion_endpoint = os.getenv("CHAT_COMPLETION_ENDPOINT")
        self.chat_completion_api_key = os.getenv("CHAT_COMPLETION_API_KEY")
        
        # Pipeline component names
        self.datasource_name = "doc-extraction-image-verbalization-ds"
        self.index_name = "doc-extraction-image-verbalization-index"
        self.skillset_name = "doc-extraction-image-verbalization-skillset"
        self.indexer_name = "doc-extraction-image-verbalization-indexer"
        
        # Initialize Azure clients
        self.credential = AzureKeyCredential(self.search_api_key)
        self.index_client = SearchIndexClient(self.search_endpoint, self.credential)
        self.indexer_client = SearchIndexerClient(self.search_endpoint, self.credential)
        self.search_client = None  # Initialized after index creation
        
        print(f"✓ Initialized Azure AI Search Pipeline")
        print(f"  Search Endpoint: {self.search_endpoint}")
        print(f"  PDF Container: {self.pdf_container}")
        print(f"  Image Container: {self.image_container}")
    
    def _validate_env(self):
        """Validate that all required environment variables are set."""
        required_vars = [
            "SEARCH_ENDPOINT",
            "SEARCH_API_KEY",
            "STORAGE_CONNECTION_STRING",
            "OPENAI_ENDPOINT",
            "OPENAI_API_KEY",
            "CHAT_COMPLETION_ENDPOINT",
            "CHAT_COMPLETION_API_KEY",
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print("Error: Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\nPlease create a .env file based on .env.template")
            sys.exit(1)
    
    def create_datasource(self) -> bool:
        """Create the Azure Blob Storage data source."""
        try:
            print(f"\n>>> Creating data source: {self.datasource_name}")
            
            datasource = SearchIndexerDataSourceConnection(
                name=self.datasource_name,
                type="azureblob",
                connection_string=self.storage_connection,
                container=SearchIndexerDataContainer(name=self.pdf_container)
            )
            
            self.indexer_client.create_or_update_data_source_connection(datasource)
            print(f"✓ Data source '{self.datasource_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error creating data source: {str(e)}")
            return False
    
    def create_index(self) -> bool:
        """Create the search index with vector search configuration."""
        try:
            print(f"\n>>> Creating search index: {self.index_name}")
            
            # Define index fields
            fields = [
                SimpleField(
                    name="content_id",
                    type=SearchFieldDataType.String,
                    key=True,
                    analyzer_name="keyword"
                ),
                SimpleField(
                    name="text_document_id",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    retrievable=True
                ),
                SearchableField(
                    name="document_title",
                    type=SearchFieldDataType.String
                ),
                SimpleField(
                    name="image_document_id",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    retrievable=True
                ),
                SearchableField(
                    name="content_text",
                    type=SearchFieldDataType.String,
                    retrievable=True
                ),
                SearchField(
                    name="content_embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=self.embedding_dimensions,
                    vector_search_profile_name="hnsw-profile",
                    retrievable=True
                ),
                SimpleField(
                    name="content_path",
                    type=SearchFieldDataType.String,
                    retrievable=True
                ),
                SimpleField(
                    name="offset",
                    type=SearchFieldDataType.String,
                    retrievable=True
                ),
                ComplexField(
                    name="location_metadata",
                    fields=[
                        SimpleField(
                            name="page_number",
                            type=SearchFieldDataType.Int32,
                            retrievable=True
                        ),
                        SimpleField(
                            name="bounding_polygons",
                            type=SearchFieldDataType.String,
                            retrievable=True
                        )
                    ]
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="hnsw-profile",
                        algorithm_configuration_name="hnsw-config",
                        vectorizer="openai-vectorizer"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "metric": "cosine"
                        }
                    )
                ],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        name="openai-vectorizer",
                        azure_open_ai_parameters=AzureOpenAIParameters(
                            resource_uri=self.openai_endpoint,
                            deployment_id=self.embedding_deployment,
                            api_key=self.openai_api_key,
                            model_name=self.embedding_model
                        )
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="document_title"),
                    content_fields=[],
                    keywords_fields=[]
                )
            )
            
            semantic_search = SemanticSearch(
                default_configuration_name="semantic-config",
                configurations=[semantic_config]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            self.index_client.create_or_update_index(index)
            print(f"✓ Index '{self.index_name}' created successfully")
            
            # Initialize search client for queries
            self.search_client = SearchClient(
                self.search_endpoint,
                self.index_name,
                self.credential
            )
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating index: {str(e)}")
            return False
    
    def create_skillset(self) -> bool:
        """Create the skillset with document extraction, text splitting, and embedding skills."""
        try:
            print(f"\n>>> Creating skillset: {self.skillset_name}")
            
            # Define skills
            skills = []
            
            # 1. Document Extraction Skill
            doc_extraction_skill = DocumentExtractionSkill(
                name="document-extraction-skill",
                description="Extract text and images from documents",
                context="/document",
                parsing_mode="default",
                data_to_extract="contentAndMetadata",
                configuration={
                    "imageAction": "generateNormalizedImages",
                    "normalizedImageMaxWidth": 2000,
                    "normalizedImageMaxHeight": 2000
                },
                inputs=[
                    InputFieldMappingEntry(name="file_data", source="/document/file_data")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="content", target_name="extracted_content"),
                    OutputFieldMappingEntry(name="normalized_images", target_name="normalized_images")
                ]
            )
            skills.append(doc_extraction_skill)
            
            # 2. Text Split Skill
            text_split_skill = SplitSkill(
                name="split-skill",
                description="Chunk documents into pages",
                context="/document",
                default_language_code="en",
                text_split_mode="pages",
                maximum_page_length=2000,
                page_overlap_length=200,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/extracted_content")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="textItems", target_name="pages")
                ]
            )
            skills.append(text_split_skill)
            
            # 3. Text Embedding Skill
            text_embedding_skill = AzureOpenAIEmbeddingSkill(
                name="text-embedding-skill",
                description="Generate embeddings for text chunks",
                context="/document/pages/*",
                resource_uri=self.openai_endpoint,
                deployment_id=self.embedding_deployment,
                api_key=self.openai_api_key,
                model_name=self.embedding_model,
                dimensions=self.embedding_dimensions,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/pages/*")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="text_vector")
                ]
            )
            skills.append(text_embedding_skill)
            
            # 4. GenAI Prompt Skill (Image Verbalization)
            system_message = """You are tasked with generating concise, accurate descriptions of images, figures, diagrams, or charts in documents. The goal is to capture the key information and meaning conveyed by the image without including extraneous details like style, colors, visual aesthetics, or size.

Instructions:
Content Focus: Describe the core content and relationships depicted in the image.
- For diagrams, specify the main elements and how they are connected or interact.
- For charts, highlight key data points, trends, comparisons, or conclusions.
- For figures or technical illustrations, identify the components and their significance.

Clarity & Precision: Use concise language to ensure clarity and technical accuracy. Avoid subjective or interpretive statements.

Avoid Visual Descriptors: Exclude details about:
- Colors, shading, and visual styles.
- Image size, layout, or decorative elements.
- Fonts, borders, and stylistic embellishments.

Context: If relevant, relate the image to the broader content of the technical document or the topic it supports.

Example Descriptions:
- Diagram: "A flowchart showing the four stages of a machine learning pipeline: data collection, preprocessing, model training, and evaluation, with arrows indicating the sequential flow of tasks."
- Chart: "A bar chart comparing the performance of four algorithms on three datasets, showing that Algorithm A consistently outperforms the others on Dataset 1."
- Figure: "A labeled diagram illustrating the components of a transformer model, including the encoder, decoder, self-attention mechanism, and feedforward layers."
"""
            
            genai_prompt_skill = WebApiSkill(
                name="genai-prompt-skill",
                description="Generate image descriptions using chat completion",
                uri=self.chat_completion_endpoint,
                timeout="PT1M",
                http_headers={"api-key": self.chat_completion_api_key},
                context="/document/normalized_images/*",
                inputs=[
                    InputFieldMappingEntry(name="systemMessage", source=f"='{system_message}'"),
                    InputFieldMappingEntry(name="userMessage", source="='Please describe this image.'"),
                    InputFieldMappingEntry(name="image", source="/document/normalized_images/*/data")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="response", target_name="verbalizedImage")
                ]
            )
            skills.append(genai_prompt_skill)
            
            # 5. Image Embedding Skill
            image_embedding_skill = AzureOpenAIEmbeddingSkill(
                name="verbalized-image-embedding-skill",
                description="Generate embeddings for verbalized images",
                context="/document/normalized_images/*",
                resource_uri=self.openai_endpoint,
                deployment_id=self.embedding_deployment,
                api_key=self.openai_api_key,
                model_name=self.embedding_model,
                dimensions=self.embedding_dimensions,
                inputs=[
                    InputFieldMappingEntry(name="text", source="/document/normalized_images/*/verbalizedImage")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="verbalizedImage_vector")
                ]
            )
            skills.append(image_embedding_skill)
            
            # 6. Shaper Skill
            shaper_skill = ShaperSkill(
                name="shaper-skill",
                description="Reshape metadata for index projection",
                context="/document/normalized_images/*",
                inputs=[
                    InputFieldMappingEntry(name="normalized_images", source="/document/normalized_images/*"),
                    InputFieldMappingEntry(
                        name="imagePath",
                        source=f"='{self.image_container}/'+$(/document/normalized_images/*/imagePath)"
                    ),
                    InputFieldMappingEntry(
                        name="location_metadata",
                        source_context="/document/normalized_images/*",
                        inputs=[
                            InputFieldMappingEntry(name="page_number", source="/document/normalized_images/*/pageNumber"),
                            InputFieldMappingEntry(name="bounding_polygons", source="/document/normalized_images/*/boundingPolygon")
                        ]
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(name="output", target_name="new_normalized_images")
                ]
            )
            skills.append(shaper_skill)
            
            # Define index projections
            index_projections = SearchIndexerIndexProjections(
                selectors=[
                    # Text projection
                    SearchIndexerIndexProjectionSelector(
                        target_index_name=self.index_name,
                        parent_key_field_name="text_document_id",
                        source_context="/document/pages/*",
                        mappings=[
                            InputFieldMappingEntry(name="content_embedding", source="/document/pages/*/text_vector"),
                            InputFieldMappingEntry(name="content_text", source="/document/pages/*"),
                            InputFieldMappingEntry(name="document_title", source="/document/document_title")
                        ]
                    ),
                    # Image projection
                    SearchIndexerIndexProjectionSelector(
                        target_index_name=self.index_name,
                        parent_key_field_name="image_document_id",
                        source_context="/document/normalized_images/*",
                        mappings=[
                            InputFieldMappingEntry(name="content_text", source="/document/normalized_images/*/verbalizedImage"),
                            InputFieldMappingEntry(name="content_embedding", source="/document/normalized_images/*/verbalizedImage_vector"),
                            InputFieldMappingEntry(name="content_path", source="/document/normalized_images/*/new_normalized_images/imagePath"),
                            InputFieldMappingEntry(name="document_title", source="/document/document_title"),
                            InputFieldMappingEntry(name="locationMetadata", source="/document/normalized_images/*/new_normalized_images/location_metadata")
                        ]
                    )
                ],
                parameters=SearchIndexerIndexProjectionsParameters(
                    projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
                )
            )
            
            # Define knowledge store
            knowledge_store = SearchIndexerKnowledgeStore(
                storage_connection_string=self.storage_connection,
                projections=[
                    SearchIndexerKnowledgeStoreProjection(
                        files=[
                            SearchIndexerKnowledgeStoreFileProjectionSelector(
                                storage_container=self.image_container,
                                source="/document/normalized_images/*"
                            )
                        ]
                    )
                ]
            )
            
            # Create skillset
            skillset = SearchIndexerSkillset(
                name=self.skillset_name,
                description="Skillset for multimodal document processing",
                skills=skills,
                index_projections=index_projections,
                knowledge_store=knowledge_store
            )
            
            self.indexer_client.create_or_update_skillset(skillset)
            print(f"✓ Skillset '{self.skillset_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error creating skillset: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_indexer(self) -> bool:
        """Create the indexer to orchestrate the pipeline."""
        try:
            print(f"\n>>> Creating indexer: {self.indexer_name}")
            
            indexer = SearchIndexer(
                name=self.indexer_name,
                data_source_name=self.datasource_name,
                target_index_name=self.index_name,
                skillset_name=self.skillset_name,
                parameters={
                    "maxFailedItems": -1,
                    "maxFailedItemsPerBatch": 0,
                    "batchSize": 1,
                    "configuration": {
                        "allowSkillsetToReadFileData": True
                    }
                },
                field_mappings=[
                    FieldMapping(
                        source_field_name="metadata_storage_name",
                        target_field_name="document_title"
                    )
                ]
            )
            
            self.indexer_client.create_or_update_indexer(indexer)
            print(f"✓ Indexer '{self.indexer_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error creating indexer: {str(e)}")
            return False
    
    def run_indexer(self) -> bool:
        """Run the indexer to process documents."""
        try:
            print(f"\n>>> Running indexer: {self.indexer_name}")
            self.indexer_client.run_indexer(self.indexer_name)
            print(f"✓ Indexer '{self.indexer_name}' started successfully")
            print("  Note: Indexing is running in the background. Use --status to check progress.")
            return True
            
        except Exception as e:
            print(f"✗ Error running indexer: {str(e)}")
            return False
    
    def get_indexer_status(self) -> bool:
        """Get the current status of the indexer."""
        try:
            print(f"\n>>> Checking indexer status: {self.indexer_name}")
            status = self.indexer_client.get_indexer_status(self.indexer_name)
            
            print(f"\nIndexer: {status.name}")
            print(f"Status: {status.status}")
            print(f"Last Result: {status.last_result.status if status.last_result else 'N/A'}")
            
            if status.last_result:
                print(f"Documents Processed: {status.last_result.item_count}")
                print(f"Documents Failed: {status.last_result.failed_item_count}")
                print(f"Start Time: {status.last_result.start_time}")
                print(f"End Time: {status.last_result.end_time}")
                
                if status.last_result.errors:
                    print(f"\nErrors ({len(status.last_result.errors)}):")
                    for error in status.last_result.errors[:5]:  # Show first 5 errors
                        print(f"  - {error.error_message}")
            
            print(f"\nExecution History ({len(status.execution_history)} runs):")
            for i, execution in enumerate(status.execution_history[:3], 1):
                print(f"  {i}. Status: {execution.status}, Items: {execution.item_count}, "
                      f"Failed: {execution.failed_item_count}, Time: {execution.start_time}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error getting indexer status: {str(e)}")
            return False
    
    def reset_indexer(self) -> bool:
        """Reset the indexer to allow full rerun."""
        try:
            print(f"\n>>> Resetting indexer: {self.indexer_name}")
            self.indexer_client.reset_indexer(self.indexer_name)
            print(f"✓ Indexer '{self.indexer_name}' reset successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error resetting indexer: {str(e)}")
            return False
    
    def search(self, query: str, top: int = 10, filter_images_only: bool = False) -> bool:
        """Execute a search query."""
        try:
            if not self.search_client:
                self.search_client = SearchClient(
                    self.search_endpoint,
                    self.index_name,
                    self.credential
                )
            
            print(f"\n>>> Searching for: '{query}'")
            
            search_params = {
                "search_text": query,
                "top": top,
                "include_total_count": True
            }
            
            if filter_images_only:
                search_params["filter"] = "image_document_id ne null"
            
            results = self.search_client.search(**search_params)
            
            print(f"\nTotal Results: {results.get_count()}\n")
            
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"  ID: {result.get('content_id', 'N/A')}")
                print(f"  Title: {result.get('document_title', 'N/A')}")
                print(f"  Content: {result.get('content_text', 'N/A')[:200]}...")
                print(f"  Image Path: {result.get('content_path', 'N/A')}")
                print(f"  Score: {result.get('@search.score', 'N/A')}")
                print()
            
            return True
            
        except Exception as e:
            print(f"✗ Error executing search: {str(e)}")
            return False
    
    def delete_all(self) -> bool:
        """Delete all pipeline components."""
        try:
            print("\n>>> Deleting all pipeline components")
            
            # Delete indexer
            try:
                self.indexer_client.delete_indexer(self.indexer_name)
                print(f"✓ Deleted indexer: {self.indexer_name}")
            except:
                print(f"  (Indexer not found: {self.indexer_name})")
            
            # Delete skillset
            try:
                self.indexer_client.delete_skillset(self.skillset_name)
                print(f"✓ Deleted skillset: {self.skillset_name}")
            except:
                print(f"  (Skillset not found: {self.skillset_name})")
            
            # Delete index
            try:
                self.index_client.delete_index(self.index_name)
                print(f"✓ Deleted index: {self.index_name}")
            except:
                print(f"  (Index not found: {self.index_name})")
            
            # Delete data source
            try:
                self.indexer_client.delete_data_source_connection(self.datasource_name)
                print(f"✓ Deleted data source: {self.datasource_name}")
            except:
                print(f"  (Data source not found: {self.datasource_name})")
            
            print("\n✓ Cleanup complete")
            return True
            
        except Exception as e:
            print(f"✗ Error during cleanup: {str(e)}")
            return False
    
    def create_all(self) -> bool:
        """Create all pipeline components."""
        print("\n" + "="*60)
        print("Creating Azure AI Search Multimodal Embedding Pipeline")
        print("="*60)
        
        success = True
        success &= self.create_datasource()
        success &= self.create_index()
        success &= self.create_skillset()
        success &= self.create_indexer()
        
        if success:
            print("\n" + "="*60)
            print("✓ Pipeline created successfully!")
            print("="*60)
            print("\nNext steps:")
            print("  1. Run the indexer: python azure_search_pipeline.py --run-indexer")
            print("  2. Check status: python azure_search_pipeline.py --status")
            print("  3. Search: python azure_search_pipeline.py --query 'your query'")
        else:
            print("\n" + "="*60)
            print("✗ Pipeline creation failed. Please check errors above.")
            print("="*60)
        
        return success


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Azure AI Search Multimodal Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all pipeline components
  python azure_search_pipeline.py --create-all
  
  # Run the indexer
  python azure_search_pipeline.py --run-indexer
  
  # Check indexer status
  python azure_search_pipeline.py --status
  
  # Search the index
  python azure_search_pipeline.py --query "energy" --top 5
  
  # Search for images only
  python azure_search_pipeline.py --query "chart" --images-only
  
  # Reset and rerun indexer
  python azure_search_pipeline.py --reset-indexer
  python azure_search_pipeline.py --run-indexer
  
  # Delete all components
  python azure_search_pipeline.py --delete-all
        """
    )
    
    parser.add_argument("--create-all", action="store_true", help="Create all pipeline components")
    parser.add_argument("--create-datasource", action="store_true", help="Create data source only")
    parser.add_argument("--create-index", action="store_true", help="Create index only")
    parser.add_argument("--create-skillset", action="store_true", help="Create skillset only")
    parser.add_argument("--create-indexer", action="store_true", help="Create indexer only")
    parser.add_argument("--run-indexer", action="store_true", help="Run the indexer")
    parser.add_argument("--reset-indexer", action="store_true", help="Reset the indexer")
    parser.add_argument("--status", action="store_true", help="Check indexer status")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top", type=int, default=10, help="Number of results to return")
    parser.add_argument("--images-only", action="store_true", help="Filter for images only")
    parser.add_argument("--delete-all", action="store_true", help="Delete all pipeline components")
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    try:
        pipeline = AzureSearchPipeline()
        
        if args.create_all:
            pipeline.create_all()
        elif args.create_datasource:
            pipeline.create_datasource()
        elif args.create_index:
            pipeline.create_index()
        elif args.create_skillset:
            pipeline.create_skillset()
        elif args.create_indexer:
            pipeline.create_indexer()
        elif args.run_indexer:
            pipeline.run_indexer()
        elif args.reset_indexer:
            pipeline.reset_indexer()
        elif args.status:
            pipeline.get_indexer_status()
        elif args.query:
            pipeline.search(args.query, args.top, args.images_only)
        elif args.delete_all:
            pipeline.delete_all()
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
