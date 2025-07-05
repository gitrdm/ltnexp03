#!/usr/bin/env python3
"""
Service Layer Comprehensive Demo
================================

Demonstrates the complete FastAPI service layer functionality including:
- REST API operations for concepts, reasoning, and frames
- Batch processing and workflow management
- WebSocket streaming capabilities
- Error handling and contract validation
- Performance monitoring and health checks

This script showcases all major features of the service layer in a
production-ready demonstration scenario.
"""

import asyncio
import aiohttp
import websockets
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sys
import subprocess
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServiceLayerDemo:
    """Comprehensive demonstration of the service layer."""
    
    def __init__(self, base_url: str = "http://localhost:8321"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.ws_base = "ws://localhost:8321/api/ws"
        self.session = None
        self.server_process = None
        
        # Demo data
        self.sample_concepts = [
            {
                "name": "knight",
                "context": "medieval warfare",
                "synset_id": "knight.n.01",
                "disambiguation": "armored warrior on horseback",
                "metadata": {"domain": "military", "era": "medieval", "nobility": True}
            },
            {
                "name": "sword",
                "context": "medieval weapons",
                "synset_id": "sword.n.01", 
                "disambiguation": "bladed melee weapon",
                "metadata": {"domain": "military", "type": "weapon", "material": "steel"}
            },
            {
                "name": "castle",
                "context": "medieval architecture",
                "synset_id": "castle.n.01",
                "disambiguation": "fortified residence",
                "metadata": {"domain": "architecture", "era": "medieval", "fortified": True}
            },
            {
                "name": "wizard",
                "context": "fantasy magic",
                "synset_id": "wizard.n.01",
                "disambiguation": "magical practitioner",
                "metadata": {"domain": "fantasy", "supernatural": True, "wisdom": True}
            },
            {
                "name": "staff",
                "context": "magical implements",
                "synset_id": "staff.n.02",
                "disambiguation": "magical rod or wand",
                "metadata": {"domain": "fantasy", "type": "implement", "magical": True}
            }
        ]
        
        self.sample_analogies = [
            {
                "source_pair": ["knight", "sword"],
                "target_pair": ["wizard", "staff"],
                "context": "fantasy character-tool relationships",
                "quality_score": 0.87,
                "reasoning": "Both represent character-primary tool pairings in fantasy settings"
            },
            {
                "source_pair": ["castle", "moat"],
                "target_pair": ["wizard", "tower"],
                "context": "defensive structures and inhabitants",
                "quality_score": 0.79,
                "reasoning": "Both represent fortified locations with characteristic defenders"
            },
            {
                "source_pair": ["sword", "steel"],
                "target_pair": ["staff", "wood"],
                "context": "implement-material relationships",
                "quality_score": 0.83,
                "reasoning": "Both represent tool-material composition relationships"
            }
        ]
    
    async def start_server(self):
        """Start the service layer server."""
        logger.info("🚀 Starting service layer server...")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, "-m", "app.service_layer"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Test if server is responding
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.api_base}/health") as response:
                        if response.status == 200:
                            logger.info("✅ Service layer server started successfully")
                            return True
                except Exception as e:
                    logger.error(f"❌ Server not responding: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the service layer server."""
        if self.server_process:
            logger.info("🛑 Stopping service layer server...")
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("✅ Server stopped")
    
    async def setup_session(self):
        """Setup HTTP session for API calls."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup_session(self):
        """Cleanup HTTP session."""
        if self.session:
            await self.session.close()
    
    async def check_health(self):
        """Check service health and status."""
        logger.info("\n📊 Checking Service Health")
        logger.info("=" * 50)
        
        try:
            # Health check
            async with self.session.get(f"{self.api_base}/health") as response:
                health_data = await response.json()
                logger.info(f"Health Status: {health_data['status']}")
                logger.info(f"Service: {health_data['service']}")
                logger.info(f"Components: {health_data['components']}")
            
            # Detailed status
            async with self.session.get(f"{self.api_base}/status") as response:
                status_data = await response.json()
                logger.info(f"Operational Status: {status_data['status']}")
                
                if 'registry_stats' in status_data:
                    stats = status_data['registry_stats']
                    logger.info(f"Registry: {stats['concepts_count']} concepts, {stats['frames_count']} frames")
                
                if 'workflow_stats' in status_data:
                    wf_stats = status_data['workflow_stats']
                    logger.info(f"Workflows: {wf_stats}")
            
            logger.info("✅ Health check completed")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    async def demo_concept_management(self):
        """Demonstrate concept management operations."""
        logger.info("\n🧠 Demonstrating Concept Management")
        logger.info("=" * 50)
        
        created_concepts = []
        
        try:
            # Create concepts
            logger.info("Creating concepts...")
            for i, concept in enumerate(self.sample_concepts):
                async with self.session.post(
                    f"{self.api_base}/concepts",
                    json=concept
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        created_concepts.append(data["concept_id"])
                        logger.info(f"  ✓ Created '{concept['name']}' (ID: {data['concept_id']})")
                    else:
                        logger.error(f"  ❌ Failed to create '{concept['name']}'")
            
            # Search concepts
            logger.info("\nSearching concepts...")
            search_data = {
                "query": "knight",
                "similarity_threshold": 0.5,
                "max_results": 5,
                "include_metadata": True
            }
            
            async with self.session.post(
                f"{self.api_base}/concepts/search",
                json=search_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"  ✓ Found {data['total_results']} concepts matching 'knight'")
                    for concept in data['concepts']:
                        logger.info(f"    - {concept['name']} (similarity: {concept['similarity_score']:.2f})")
            
            # Compute similarity
            logger.info("\nComputing concept similarity...")
            similarity_data = {
                "concept1": "knight",
                "concept2": "sword",
                "similarity_method": "hybrid"
            }
            
            async with self.session.post(
                f"{self.api_base}/concepts/similarity",
                json=similarity_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"  ✓ Similarity between 'knight' and 'sword': {data['similarity_score']:.3f}")
                    logger.info(f"    Method: {data['method_used']}, Confidence: {data['confidence']:.2f}")
            
            # Retrieve individual concepts
            logger.info("\nRetrieving individual concepts...")
            for concept_id in created_concepts[:2]:  # Just show first 2
                async with self.session.get(f"{self.api_base}/concepts/{concept_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"  ✓ Retrieved '{data['name']}' - {data.get('disambiguation', 'N/A')}")
            
            logger.info("✅ Concept management demo completed")
            return created_concepts
            
        except Exception as e:
            logger.error(f"❌ Concept management demo failed: {e}")
            return []
    
    async def demo_semantic_reasoning(self):
        """Demonstrate semantic reasoning capabilities."""
        logger.info("\n🧩 Demonstrating Semantic Reasoning")
        logger.info("=" * 50)
        
        try:
            # Analogy completion
            logger.info("Completing analogies...")
            analogy_data = {
                "source_a": "knight",
                "source_b": "sword",
                "target_a": "wizard",
                "context": "fantasy",
                "max_completions": 3,
                "min_confidence": 0.4
            }
            
            async with self.session.post(
                f"{self.api_base}/analogies/complete",
                json=analogy_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"  ✓ Analogy: knight:sword :: wizard:?")
                    for i, completion in enumerate(data['completions']):
                        target_b = completion.get('target_b', 'unknown')
                        confidence = completion.get('confidence', 0.0)
                        logger.info(f"    {i+1}. '{target_b}' (confidence: {confidence:.2f})")
            
            # Semantic field discovery
            logger.info("\nDiscovering semantic fields...")
            discovery_data = {
                "domain": "medieval",
                "min_coherence": 0.6,
                "max_fields": 3,
                "clustering_method": "kmeans"
            }
            
            async with self.session.post(
                f"{self.api_base}/semantic-fields/discover",
                json=discovery_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"  ✓ Discovered {len(data['semantic_fields'])} semantic fields")
                    for field in data['semantic_fields']:
                        field_name = field.get('name', 'Unknown')
                        coherence = field.get('coherence', 0.0)
                        logger.info(f"    - '{field_name}' (coherence: {coherence:.2f})")
            
            # Cross-domain analogies
            logger.info("\nDiscovering cross-domain analogies...")
            cross_domain_data = {
                "source_domain": "medieval",
                "target_domain": "fantasy",
                "min_quality": 0.6,
                "max_analogies": 3
            }
            
            async with self.session.post(
                f"{self.api_base}/analogies/cross-domain",
                json=cross_domain_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"  ✓ Found {len(data['analogies'])} cross-domain analogies")
                    for analogy in data['analogies']:
                        source_pair = analogy.get('source_pair', [])
                        target_pair = analogy.get('target_pair', [])
                        quality = analogy.get('quality_score', 0.0)
                        logger.info(f"    - {source_pair} ↔ {target_pair} (quality: {quality:.2f})")
            
            logger.info("✅ Semantic reasoning demo completed")
            
        except Exception as e:
            logger.error(f"❌ Semantic reasoning demo failed: {e}")
    
    async def demo_frame_operations(self):
        """Demonstrate frame creation and management."""
        logger.info("\n🖼️ Demonstrating Frame Operations")
        logger.info("=" * 50)
        
        try:
            # Create frame
            logger.info("Creating semantic frame...")
            frame_data = {
                "name": "Combat",
                "definition": "A situation involving conflict between entities using weapons",
                "core_elements": ["Combatant_1", "Combatant_2", "Weapon", "Location"],
                "peripheral_elements": ["Time", "Outcome", "Audience"],
                "lexical_units": ["fight", "battle", "duel", "combat", "clash"],
                "metadata": {"domain": "conflict", "complexity": "high"}
            }
            
            async with self.session.post(
                f"{self.api_base}/frames",
                json=frame_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    frame_id = data["frame_id"]
                    logger.info(f"  ✓ Created frame '{data['name']}' (ID: {frame_id})")
                    logger.info(f"    Definition: {data['definition']}")
                    
                    # Create frame instance
                    logger.info("\nCreating frame instance...")
                    instance_data = {
                        "instance_id": "knight_vs_dragon_001",
                        "concept_bindings": {
                            "Combatant_1": "knight",
                            "Combatant_2": "dragon",
                            "Weapon": "sword",
                            "Location": "castle courtyard"
                        },
                        "context": "medieval fantasy battle scenario",
                        "confidence": 0.92
                    }
                    
                    async with self.session.post(
                        f"{self.api_base}/frames/{frame_id}/instances",
                        json=instance_data
                    ) as inst_response:
                        if inst_response.status == 200:
                            inst_data = await inst_response.json()
                            logger.info(f"  ✓ Created instance '{inst_data['instance_id']}'")
                            logger.info(f"    Bindings: {inst_data['bindings']}")
                    
                    # Query frames
                    logger.info("\nQuerying frames...")
                    query_data = {
                        "concept": "combat",
                        "max_results": 5
                    }
                    
                    async with self.session.post(
                        f"{self.api_base}/frames/query",
                        json=query_data
                    ) as query_response:
                        if query_response.status == 200:
                            query_result = await query_response.json()
                            logger.info(f"  ✓ Found {len(query_result['frames'])} frames matching 'combat'")
                            logger.info(f"    Found {len(query_result['instances'])} instances")
            
            logger.info("✅ Frame operations demo completed")
            
        except Exception as e:
            logger.error(f"❌ Frame operations demo failed: {e}")
    
    async def demo_batch_operations(self):
        """Demonstrate batch processing and workflows."""
        logger.info("\n📦 Demonstrating Batch Operations")
        logger.info("=" * 50)
        
        try:
            # Create analogy batch
            logger.info("Creating analogy batch...")
            batch_data = {
                "analogies": self.sample_analogies,
                "workflow_id": "demo_workflow_001"
            }
            
            async with self.session.post(
                f"{self.api_base}/batch/analogies",
                json=batch_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    workflow_id = data["workflow_id"]
                    logger.info(f"  ✓ Created workflow '{workflow_id}'")
                    logger.info(f"    Type: {data['workflow_type']}")
                    logger.info(f"    Status: {data['status']}")
                    logger.info(f"    Total items: {data['items_total']}")
                    
                    # Wait a moment for processing
                    await asyncio.sleep(1)
                    
                    # Check workflow status
                    logger.info("\nChecking workflow status...")
                    async with self.session.get(
                        f"{self.api_base}/batch/workflows/{workflow_id}"
                    ) as status_response:
                        if status_response.status == 200:
                            status_data = await status_response.json()
                            logger.info(f"  ✓ Workflow status: {status_data['status']}")
                            logger.info(f"    Processed: {status_data['items_processed']}/{status_data['items_total']}")
                            logger.info(f"    Errors: {status_data['error_count']}")
                    
                    # List all workflows
                    logger.info("\nListing all workflows...")
                    async with self.session.get(
                        f"{self.api_base}/batch/workflows"
                    ) as list_response:
                        if list_response.status == 200:
                            workflows = await list_response.json()
                            logger.info(f"  ✓ Found {len(workflows)} total workflows")
                            for workflow in workflows:
                                wf_id = workflow['workflow_id']
                                wf_status = workflow['status']
                                wf_type = workflow['workflow_type']
                                logger.info(f"    - {wf_id}: {wf_type} ({wf_status})")
            
            logger.info("✅ Batch operations demo completed")
            
        except Exception as e:
            logger.error(f"❌ Batch operations demo failed: {e}")
    
    async def demo_websocket_streaming(self):
        """Demonstrate WebSocket streaming capabilities."""
        logger.info("\n🌊 Demonstrating WebSocket Streaming")
        logger.info("=" * 50)
        
        try:
            # Stream analogies
            logger.info("Streaming analogies via WebSocket...")
            
            uri = f"{self.ws_base}/analogies/stream?min_quality=0.5"
            
            async with websockets.connect(uri) as websocket:
                logger.info("  ✓ WebSocket connected")
                
                # Receive messages
                message_count = 0
                timeout_count = 0
                max_timeout = 3
                
                while timeout_count < max_timeout:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "connection":
                            logger.info(f"    Connected: {data['message']}")
                        elif data.get("message_type") == "analogy":
                            content = data.get("content", {})
                            analogy = content.get("analogy", {})
                            count = content.get("count", message_count)
                            logger.info(f"    Received analogy #{count}: {analogy}")
                            message_count += 1
                        elif data.get("type") == "completion":
                            logger.info(f"    Streaming completed: {data['total_streamed']} items")
                            break
                        elif data.get("type") == "error":
                            logger.error(f"    Stream error: {data['error']}")
                            break
                        
                        timeout_count = 0  # Reset timeout counter
                        
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        if timeout_count >= max_timeout:
                            logger.info("    No more messages, closing stream")
                            break
                    except Exception as e:
                        logger.error(f"    WebSocket error: {e}")
                        break
                
                logger.info(f"  ✓ Received {message_count} analogy messages")
            
            logger.info("✅ WebSocket streaming demo completed")
            
        except Exception as e:
            logger.error(f"❌ WebSocket streaming demo failed: {e}")
    
    async def demo_error_handling(self):
        """Demonstrate error handling and validation."""
        logger.info("\n⚠️ Demonstrating Error Handling")
        logger.info("=" * 50)
        
        try:
            # Invalid concept creation
            logger.info("Testing invalid concept creation...")
            invalid_concept = {
                "name": "",  # Empty name should fail
                "context": "test"
            }
            
            async with self.session.post(
                f"{self.api_base}/concepts",
                json=invalid_concept
            ) as response:
                if response.status == 422:
                    logger.info("  ✓ Correctly rejected invalid concept (empty name)")
                else:
                    logger.warning(f"  ⚠️ Unexpected status: {response.status}")
            
            # Non-existent concept retrieval
            logger.info("Testing non-existent concept retrieval...")
            async with self.session.get(
                f"{self.api_base}/concepts/nonexistent_concept_id"
            ) as response:
                if response.status == 404:
                    logger.info("  ✓ Correctly returned 404 for non-existent concept")
                else:
                    logger.warning(f"  ⚠️ Unexpected status: {response.status}")
            
            # Invalid similarity computation
            logger.info("Testing invalid similarity computation...")
            invalid_similarity = {
                "concept1": "nonexistent_concept_1",
                "concept2": "nonexistent_concept_2",
                "similarity_method": "hybrid"
            }
            
            async with self.session.post(
                f"{self.api_base}/concepts/similarity",
                json=invalid_similarity
            ) as response:
                if response.status in [404, 400]:
                    logger.info("  ✓ Correctly rejected similarity for non-existent concepts")
                else:
                    logger.warning(f"  ⚠️ Unexpected status: {response.status}")
            
            # Invalid workflow retrieval
            logger.info("Testing non-existent workflow retrieval...")
            async with self.session.get(
                f"{self.api_base}/batch/workflows/nonexistent_workflow"
            ) as response:
                if response.status == 404:
                    logger.info("  ✓ Correctly returned 404 for non-existent workflow")
                else:
                    logger.warning(f"  ⚠️ Unexpected status: {response.status}")
            
            logger.info("✅ Error handling demo completed")
            
        except Exception as e:
            logger.error(f"❌ Error handling demo failed: {e}")
    
    async def performance_analysis(self):
        """Analyze performance characteristics."""
        logger.info("\n⚡ Performance Analysis")
        logger.info("=" * 50)
        
        try:
            # Concept creation performance
            logger.info("Testing concept creation performance...")
            start_time = time.time()
            
            test_concepts = []
            for i in range(10):
                concept = {
                    "name": f"perf_test_concept_{i}",
                    "context": "performance_testing",
                    "metadata": {"test_id": i, "batch": "performance"}
                }
                test_concepts.append(concept)
            
            # Create concepts sequentially
            created_count = 0
            for concept in test_concepts:
                async with self.session.post(
                    f"{self.api_base}/concepts",
                    json=concept
                ) as response:
                    if response.status == 200:
                        created_count += 1
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = created_count / duration if duration > 0 else 0
            
            logger.info(f"  ✓ Created {created_count}/10 concepts")
            logger.info(f"    Duration: {duration:.3f} seconds")
            logger.info(f"    Throughput: {throughput:.2f} concepts/second")
            
            # Batch operation performance
            logger.info("\nTesting batch operation performance...")
            large_batch = self.sample_analogies * 5  # 15 analogies
            
            batch_start = time.time()
            async with self.session.post(
                f"{self.api_base}/batch/analogies",
                json={"analogies": large_batch, "workflow_id": "perf_test_batch"}
            ) as response:
                batch_end = time.time()
                
                if response.status == 200:
                    batch_duration = batch_end - batch_start
                    logger.info(f"  ✓ Created batch with {len(large_batch)} analogies")
                    logger.info(f"    Batch creation time: {batch_duration:.3f} seconds")
                    logger.info(f"    Rate: {len(large_batch)/batch_duration:.2f} analogies/second")
            
            logger.info("✅ Performance analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
    
    async def run_complete_demo(self):
        """Run the complete service layer demonstration."""
        logger.info("🎬 Starting Complete Service Layer Demo")
        logger.info("=" * 70)
        
        try:
            # Setup
            await self.setup_session()
            
            # Run all demonstrations
            await self.check_health()
            concept_ids = await self.demo_concept_management()
            await self.demo_semantic_reasoning()
            await self.demo_frame_operations()
            await self.demo_batch_operations()
            await self.demo_websocket_streaming()
            await self.demo_error_handling()
            await self.performance_analysis()
            
            # Summary
            logger.info("\n🎉 Demo Summary")
            logger.info("=" * 50)
            logger.info(f"✅ Created {len(concept_ids)} concepts")
            logger.info("✅ Demonstrated semantic reasoning")
            logger.info("✅ Demonstrated frame operations")
            logger.info("✅ Demonstrated batch processing")
            logger.info("✅ Demonstrated WebSocket streaming")
            logger.info("✅ Demonstrated error handling")
            logger.info("✅ Completed performance analysis")
            
            logger.info("\n🚀 Service Layer Demo Completed Successfully!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise
        finally:
            await self.cleanup_session()


async def main():
    """Main demo execution function."""
    demo = ServiceLayerDemo()
    
    try:
        # Start server
        server_started = await demo.start_server()
        if not server_started:
            logger.error("❌ Failed to start server. Exiting.")
            return
        
        # Run demo
        await demo.run_complete_demo()
        
    except KeyboardInterrupt:
        logger.info("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
    finally:
        # Stop server
        await demo.stop_server()


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("app/service_layer.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Usage: python demo_service_layer.py")
        sys.exit(1)
    
    # Install required packages if needed
    try:
        import aiohttp
        import websockets
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "websockets"])
        import aiohttp
        import websockets
    
    # Run the demo
    asyncio.run(main())
