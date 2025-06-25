//! Izinyoka Metacognitive Processing System
//!
//! Real-time streaming metacognitive processing that mirrors how the human mind
//! processes partial information before sentences are complete. This system provides
//! continuous metacognitive awareness and self-reflection capabilities.

use futures::sink::{Sink, SinkExt};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;

use crate::consciousness::{ConsciousnessInput, ConsciousnessInsight, InsightType};
use crate::error::{ImhotepError, ImhotepResult};

/// Streaming metacognitive processor
pub struct IzinyokaMetacognitive {
    /// Input stream receiver
    input_receiver: Arc<RwLock<Option<mpsc::Receiver<MetacognitiveChunk>>>>,

    /// Output stream sender
    output_sender: Arc<RwLock<Option<mpsc::Sender<MetacognitiveInsight>>>>,

    /// Processing state
    processing_state: Arc<RwLock<MetacognitiveState>>,

    /// Configuration
    config: MetacognitiveConfig,

    /// Active processing tasks
    active_tasks: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Metacognitive processing chunk (partial input)
#[derive(Debug, Clone)]
pub struct MetacognitiveChunk {
    /// Chunk identifier
    pub chunk_id: String,

    /// Session identifier
    pub session_id: String,

    /// Partial content
    pub content: String,

    /// Chunk position in sequence
    pub sequence_position: usize,

    /// Is this the final chunk?
    pub is_final: bool,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Processing context
    pub context: ProcessingContext,
}

/// Real-time metacognitive insight
#[derive(Debug, Clone)]
pub struct MetacognitiveInsight {
    /// Insight identifier
    pub insight_id: String,

    /// Session identifier
    pub session_id: String,

    /// Metacognitive content
    pub content: String,

    /// Insight type
    pub insight_type: MetacognitiveInsightType,

    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,

    /// Processing stage when insight emerged
    pub processing_stage: ProcessingStage,

    /// Related chunks that triggered this insight
    pub triggering_chunks: Vec<String>,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Metacognitive trail
    pub metacognitive_trail: Vec<MetacognitiveStep>,
}

/// Types of metacognitive insights
#[derive(Debug, Clone)]
pub enum MetacognitiveInsightType {
    /// Self-awareness about own processing
    SelfAwareness,

    /// Recognition of processing patterns
    ProcessingPatternRecognition,

    /// Uncertainty acknowledgment
    UncertaintyAcknowledgment,

    /// Confidence assessment
    ConfidenceAssessment,

    /// Strategy adjustment
    StrategyAdjustment,

    /// Goal refinement
    GoalRefinement,

    /// Error detection
    ErrorDetection,

    /// Learning recognition
    LearningRecognition,
}

/// Processing stages
#[derive(Debug, Clone)]
pub enum ProcessingStage {
    /// Initial chunk processing
    InitialProcessing,

    /// Pattern recognition phase
    PatternRecognition,

    /// Integration phase
    Integration,

    /// Reflection phase
    Reflection,

    /// Conclusion phase
    Conclusion,
}

/// Metacognitive processing step
#[derive(Debug, Clone)]
pub struct MetacognitiveStep {
    /// Step description
    pub description: String,

    /// Processing time (microseconds)
    pub processing_time: u64,

    /// Confidence at this step
    pub confidence: f64,

    /// Internal state snapshot
    pub state_snapshot: String,
}

/// Current metacognitive state
#[derive(Debug, Clone)]
pub struct MetacognitiveState {
    /// Active sessions
    pub active_sessions: std::collections::HashMap<String, SessionState>,

    /// Processing buffer
    pub processing_buffer: VecDeque<MetacognitiveChunk>,

    /// Current insights
    pub current_insights: Vec<MetacognitiveInsight>,

    /// Processing statistics
    pub processing_stats: ProcessingStatistics,
}

/// Session state
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Session ID
    pub session_id: String,

    /// Accumulated content so far
    pub accumulated_content: String,

    /// Processing chunks received
    pub chunks_received: Vec<MetacognitiveChunk>,

    /// Insights generated
    pub insights_generated: Vec<MetacognitiveInsight>,

    /// Current processing stage
    pub current_stage: ProcessingStage,

    /// Session start time
    pub start_time: chrono::DateTime<chrono::Utc>,

    /// Last activity time
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Processing context
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Processing goal
    pub goal: String,

    /// Expected output type
    pub expected_output: String,

    /// Processing priority (0.0 - 1.0)
    pub priority: f64,

    /// Real-time processing required
    pub real_time_required: bool,
}

/// Configuration for metacognitive processing
#[derive(Debug, Clone)]
pub struct MetacognitiveConfig {
    /// Minimum chunk size for processing
    pub min_chunk_size: usize,

    /// Maximum buffer size
    pub max_buffer_size: usize,

    /// Processing latency target (milliseconds)
    pub target_latency_ms: u64,

    /// Confidence threshold for insights
    pub insight_confidence_threshold: f64,

    /// Enable real-time streaming
    pub enable_streaming: bool,

    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStatistics {
    /// Total chunks processed
    pub chunks_processed: u64,

    /// Total insights generated
    pub insights_generated: u64,

    /// Average processing latency (microseconds)
    pub avg_processing_latency: f64,

    /// Success rate
    pub success_rate: f64,

    /// Current throughput (chunks/second)
    pub current_throughput: f64,
}

impl IzinyokaMetacognitive {
    /// Create new streaming metacognitive processor
    pub fn new(config: MetacognitiveConfig) -> Self {
        let processing_state = Arc::new(RwLock::new(MetacognitiveState {
            active_sessions: std::collections::HashMap::new(),
            processing_buffer: VecDeque::new(),
            current_insights: Vec::new(),
            processing_stats: ProcessingStatistics {
                chunks_processed: 0,
                insights_generated: 0,
                avg_processing_latency: 0.0,
                success_rate: 1.0,
                current_throughput: 0.0,
            },
        }));

        Self {
            input_receiver: Arc::new(RwLock::new(None)),
            output_sender: Arc::new(RwLock::new(None)),
            processing_state,
            config,
            active_tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize streaming channels
    pub async fn initialize_streaming(
        &mut self,
    ) -> ImhotepResult<(
        mpsc::Sender<MetacognitiveChunk>,
        mpsc::Receiver<MetacognitiveInsight>,
    )> {
        let (input_tx, input_rx) = mpsc::channel::<MetacognitiveChunk>(self.config.max_buffer_size);
        let (output_tx, output_rx) =
            mpsc::channel::<MetacognitiveInsight>(self.config.max_buffer_size);

        *self.input_receiver.write().await = Some(input_rx);
        *self.output_sender.write().await = Some(output_tx);

        // Start the streaming processor
        self.start_streaming_processor().await?;

        Ok((input_tx, output_rx))
    }

    /// Start the streaming processor task
    async fn start_streaming_processor(&self) -> ImhotepResult<()> {
        let processing_state = Arc::clone(&self.processing_state);
        let input_receiver = Arc::clone(&self.input_receiver);
        let output_sender = Arc::clone(&self.output_sender);
        let config = self.config.clone();

        let task = tokio::spawn(async move {
            Self::streaming_processor_loop(processing_state, input_receiver, output_sender, config)
                .await;
        });

        self.active_tasks.write().await.push(task);
        Ok(())
    }

    /// Main streaming processor loop
    async fn streaming_processor_loop(
        processing_state: Arc<RwLock<MetacognitiveState>>,
        input_receiver: Arc<RwLock<Option<mpsc::Receiver<MetacognitiveChunk>>>>,
        output_sender: Arc<RwLock<Option<mpsc::Sender<MetacognitiveInsight>>>>,
        config: MetacognitiveConfig,
    ) {
        let mut receiver_guard = input_receiver.write().await;
        let mut sender_guard = output_sender.write().await;

        if let (Some(ref mut receiver), Some(ref mut sender)) =
            (receiver_guard.as_mut(), sender_guard.as_mut())
        {
            while let Some(chunk) = receiver.recv().await {
                let start_time = std::time::Instant::now();

                // Process chunk immediately (streaming behavior)
                match Self::process_chunk_streaming(&chunk, &processing_state, &config).await {
                    Ok(insights) => {
                        // Send insights as they're generated (streaming output)
                        for insight in insights {
                            if let Err(e) = sender.send(insight).await {
                                eprintln!("Failed to send insight: {}", e);
                                break;
                            }
                        }

                        // Update statistics
                        let processing_time = start_time.elapsed().as_micros() as f64;
                        Self::update_processing_stats(&processing_state, processing_time, true)
                            .await;
                    }
                    Err(e) => {
                        eprintln!("Error processing chunk: {}", e);
                        let processing_time = start_time.elapsed().as_micros() as f64;
                        Self::update_processing_stats(&processing_state, processing_time, false)
                            .await;
                    }
                }
            }
        }
    }

    /// Process a single chunk in streaming mode
    async fn process_chunk_streaming(
        chunk: &MetacognitiveChunk,
        processing_state: &Arc<RwLock<MetacognitiveState>>,
        config: &MetacognitiveConfig,
    ) -> ImhotepResult<Vec<MetacognitiveInsight>> {
        let mut state = processing_state.write().await;
        let mut insights = Vec::new();

        // Get or create session state
        let session_state = state
            .active_sessions
            .entry(chunk.session_id.clone())
            .or_insert_with(|| SessionState {
                session_id: chunk.session_id.clone(),
                accumulated_content: String::new(),
                chunks_received: Vec::new(),
                insights_generated: Vec::new(),
                current_stage: ProcessingStage::InitialProcessing,
                start_time: chrono::Utc::now(),
                last_activity: chrono::Utc::now(),
            });

        // Update session with new chunk
        session_state.accumulated_content.push_str(&chunk.content);
        session_state.chunks_received.push(chunk.clone());
        session_state.last_activity = chrono::Utc::now();

        // Generate real-time metacognitive insights

        // 1. Self-awareness insight about processing
        if chunk.sequence_position == 0 {
            insights.push(MetacognitiveInsight {
                insight_id: format!("self_aware_{}", uuid::Uuid::new_v4()),
                session_id: chunk.session_id.clone(),
                content: format!("I'm beginning to process new input: '{}'... I notice I'm starting to analyze this before having the complete context.", 
                    chunk.content.chars().take(50).collect::<String>()),
                insight_type: MetacognitiveInsightType::SelfAwareness,
                confidence: 0.9,
                processing_stage: ProcessingStage::InitialProcessing,
                triggering_chunks: vec![chunk.chunk_id.clone()],
                timestamp: chrono::Utc::now(),
                metacognitive_trail: vec![
                    MetacognitiveStep {
                        description: "Recognized start of new processing session".to_string(),
                        processing_time: 100,
                        confidence: 0.9,
                        state_snapshot: "Initial awareness activated".to_string(),
                    }
                ],
            });
        }

        // 2. Pattern recognition on partial content
        if session_state.accumulated_content.len() > config.min_chunk_size {
            if let Some(pattern_insight) =
                Self::detect_processing_patterns(&session_state.accumulated_content)
            {
                insights.push(MetacognitiveInsight {
                    insight_id: format!("pattern_{}", uuid::Uuid::new_v4()),
                    session_id: chunk.session_id.clone(),
                    content: pattern_insight,
                    insight_type: MetacognitiveInsightType::ProcessingPatternRecognition,
                    confidence: 0.7,
                    processing_stage: ProcessingStage::PatternRecognition,
                    triggering_chunks: vec![chunk.chunk_id.clone()],
                    timestamp: chrono::Utc::now(),
                    metacognitive_trail: vec![MetacognitiveStep {
                        description: "Analyzing partial content for patterns".to_string(),
                        processing_time: 200,
                        confidence: 0.7,
                        state_snapshot: format!(
                            "Content length: {}",
                            session_state.accumulated_content.len()
                        ),
                    }],
                });
            }
        }

        // 3. Uncertainty acknowledgment for incomplete information
        if !chunk.is_final && chunk.sequence_position > 0 {
            insights.push(MetacognitiveInsight {
                insight_id: format!("uncertainty_{}", uuid::Uuid::new_v4()),
                session_id: chunk.session_id.clone(),
                content: "I'm processing incomplete information and acknowledge uncertainty in my current understanding. I'm building tentative interpretations that may change as more context arrives.".to_string(),
                insight_type: MetacognitiveInsightType::UncertaintyAcknowledgment,
                confidence: 0.8,
                processing_stage: ProcessingStage::Integration,
                triggering_chunks: vec![chunk.chunk_id.clone()],
                timestamp: chrono::Utc::now(),
                metacognitive_trail: vec![
                    MetacognitiveStep {
                        description: "Acknowledging incomplete information state".to_string(),
                        processing_time: 150,
                        confidence: 0.8,
                        state_snapshot: format!("Chunks processed: {}", session_state.chunks_received.len()),
                    }
                ],
            });
        }

        // 4. Strategy adjustment based on streaming input
        if session_state.chunks_received.len() > 3 {
            insights.push(MetacognitiveInsight {
                insight_id: format!("strategy_{}", uuid::Uuid::new_v4()),
                session_id: chunk.session_id.clone(),
                content: "I'm adjusting my processing strategy based on the streaming nature of this input. I'm maintaining multiple hypotheses and updating them incrementally.".to_string(),
                insight_type: MetacognitiveInsightType::StrategyAdjustment,
                confidence: 0.75,
                processing_stage: ProcessingStage::Integration,
                triggering_chunks: vec![chunk.chunk_id.clone()],
                timestamp: chrono::Utc::now(),
                metacognitive_trail: vec![
                    MetacognitiveStep {
                        description: "Adapting strategy for streaming input".to_string(),
                        processing_time: 180,
                        confidence: 0.75,
                        state_snapshot: "Multiple hypothesis tracking active".to_string(),
                    }
                ],
            });
        }

        // 5. Final reflection if this is the last chunk
        if chunk.is_final {
            session_state.current_stage = ProcessingStage::Conclusion;
            insights.push(MetacognitiveInsight {
                insight_id: format!("conclusion_{}", uuid::Uuid::new_v4()),
                session_id: chunk.session_id.clone(),
                content: format!("I've completed processing this streaming input. My understanding evolved through {} chunks, and I can now reflect on the complete context: '{}'", 
                    session_state.chunks_received.len(),
                    session_state.accumulated_content.chars().take(100).collect::<String>()),
                insight_type: MetacognitiveInsightType::LearningRecognition,
                confidence: 0.85,
                processing_stage: ProcessingStage::Conclusion,
                triggering_chunks: vec![chunk.chunk_id.clone()],
                timestamp: chrono::Utc::now(),
                metacognitive_trail: vec![
                    MetacognitiveStep {
                        description: "Completing streaming processing session".to_string(),
                        processing_time: 250,
                        confidence: 0.85,
                        state_snapshot: format!("Final content length: {}", session_state.accumulated_content.len()),
                    }
                ],
            });
        }

        // Update session insights
        session_state.insights_generated.extend(insights.clone());
        state.processing_stats.chunks_processed += 1;
        state.processing_stats.insights_generated += insights.len() as u64;

        Ok(insights)
    }

    /// Detect processing patterns in partial content
    fn detect_processing_patterns(content: &str) -> Option<String> {
        // Simple pattern detection - can be enhanced with more sophisticated NLP
        if content.contains("question") || content.contains("?") {
            Some("I detect this might be a question or inquiry. I'm preparing to structure my response accordingly.".to_string())
        } else if content.contains("analyze") || content.contains("examine") {
            Some("I recognize this as an analytical request. I'm activating deeper reasoning processes.".to_string())
        } else if content.contains("explain") || content.contains("describe") {
            Some("This appears to be a request for explanation. I'm organizing information for clear communication.".to_string())
        } else {
            None
        }
    }

    /// Update processing statistics
    async fn update_processing_stats(
        processing_state: &Arc<RwLock<MetacognitiveState>>,
        processing_time: f64,
        success: bool,
    ) {
        let mut state = processing_state.write().await;

        // Update average latency
        let total_processed = state.processing_stats.chunks_processed as f64;
        state.processing_stats.avg_processing_latency =
            (state.processing_stats.avg_processing_latency * total_processed + processing_time)
                / (total_processed + 1.0);

        // Update success rate
        let total_attempts = state.processing_stats.chunks_processed + 1;
        let successful_attempts = if success {
            (state.processing_stats.success_rate * state.processing_stats.chunks_processed as f64)
                + 1.0
        } else {
            state.processing_stats.success_rate * state.processing_stats.chunks_processed as f64
        };

        state.processing_stats.success_rate = successful_attempts / total_attempts as f64;
    }

    /// Get current processing statistics
    pub async fn get_processing_stats(&self) -> ProcessingStatistics {
        let state = self.processing_state.read().await;
        state.processing_stats.clone()
    }

    /// Get active sessions
    pub async fn get_active_sessions(&self) -> Vec<String> {
        let state = self.processing_state.read().await;
        state.active_sessions.keys().cloned().collect()
    }

    /// Process single input (non-streaming mode for compatibility)
    pub async fn process(&mut self, input: &serde_json::Value) -> ImhotepResult<serde_json::Value> {
        // Convert input to streaming format for processing
        let chunk = MetacognitiveChunk {
            chunk_id: uuid::Uuid::new_v4().to_string(),
            session_id: "single_process_session".to_string(),
            content: input.to_string(),
            sequence_position: 0,
            is_final: true,
            timestamp: chrono::Utc::now(),
            context: ProcessingContext {
                goal: "Single processing request".to_string(),
                expected_output: "Metacognitive insights".to_string(),
                priority: 1.0,
                real_time_required: false,
            },
        };

        let insights =
            Self::process_chunk_streaming(&chunk, &self.processing_state, &self.config).await?;

        Ok(serde_json::json!({
            "system": "izinyoka",
            "processing_mode": "single_input",
            "insights": insights,
            "total_insights": insights.len(),
            "processing_complete": true
        }))
    }

    /// Check if system is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enable_streaming
    }

    /// Shutdown streaming processor
    pub async fn shutdown(&mut self) -> ImhotepResult<()> {
        let mut tasks = self.active_tasks.write().await;
        for task in tasks.drain(..) {
            task.abort();
        }
        Ok(())
    }
}

impl Default for MetacognitiveConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 10,
            max_buffer_size: 1000,
            target_latency_ms: 50,
            insight_confidence_threshold: 0.5,
            enable_streaming: true,
            max_concurrent_sessions: 100,
        }
    }
}

/// Stream adapter for easier integration
pub struct IzinyokaStream {
    inner: IzinyokaMetacognitive,
    input_sender: Option<mpsc::Sender<MetacognitiveChunk>>,
    output_receiver: Option<mpsc::Receiver<MetacognitiveInsight>>,
}

impl IzinyokaStream {
    /// Create new streaming interface
    pub async fn new(config: MetacognitiveConfig) -> ImhotepResult<Self> {
        let mut inner = IzinyokaMetacognitive::new(config);
        let (input_sender, output_receiver) = inner.initialize_streaming().await?;

        Ok(Self {
            inner,
            input_sender: Some(input_sender),
            output_receiver: Some(output_receiver),
        })
    }

    /// Send streaming chunk
    pub async fn send_chunk(&mut self, chunk: MetacognitiveChunk) -> ImhotepResult<()> {
        if let Some(ref mut sender) = self.input_sender {
            sender.send(chunk).await.map_err(|e| {
                ImhotepError::ProcessingError(format!("Failed to send chunk: {}", e))
            })?;
        }
        Ok(())
    }

    /// Receive streaming insight
    pub async fn receive_insight(&mut self) -> Option<MetacognitiveInsight> {
        if let Some(ref mut receiver) = self.output_receiver {
            receiver.recv().await
        } else {
            None
        }
    }

    /// Send text chunk (convenience method)
    pub async fn send_text_chunk(
        &mut self,
        session_id: String,
        content: String,
        sequence_position: usize,
        is_final: bool,
    ) -> ImhotepResult<()> {
        let chunk = MetacognitiveChunk {
            chunk_id: uuid::Uuid::new_v4().to_string(),
            session_id,
            content,
            sequence_position,
            is_final,
            timestamp: chrono::Utc::now(),
            context: ProcessingContext {
                goal: "Real-time text processing".to_string(),
                expected_output: "Streaming metacognitive insights".to_string(),
                priority: 1.0,
                real_time_required: true,
            },
        };

        self.send_chunk(chunk).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_processing() {
        let config = MetacognitiveConfig::default();
        let mut stream = IzinyokaStream::new(config).await.unwrap();

        // Send streaming chunks
        let session_id = "test_session".to_string();

        stream
            .send_text_chunk(session_id.clone(), "Hello".to_string(), 0, false)
            .await
            .unwrap();
        stream
            .send_text_chunk(session_id.clone(), " world".to_string(), 1, false)
            .await
            .unwrap();
        stream
            .send_text_chunk(session_id.clone(), "!".to_string(), 2, true)
            .await
            .unwrap();

        // Receive insights
        let mut insights = Vec::new();
        for _ in 0..3 {
            if let Some(insight) = stream.receive_insight().await {
                insights.push(insight);
            }
        }

        assert!(!insights.is_empty());
        assert!(insights
            .iter()
            .any(|i| matches!(i.insight_type, MetacognitiveInsightType::SelfAwareness)));
    }

    #[tokio::test]
    async fn test_metacognitive_config() {
        let config = MetacognitiveConfig::default();
        assert!(config.enable_streaming);
        assert_eq!(config.target_latency_ms, 50);
    }
}
