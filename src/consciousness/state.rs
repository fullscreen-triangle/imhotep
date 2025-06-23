//! Consciousness State Module
//! 
//! This module manages the current state of consciousness using BMD principles,
//! tracking consciousness evolution, state transitions, and maintaining state coherence
//! through Mizraji's information catalysis framework.

use std::collections::HashMap;
use crate::error::{ImhotepError, ImhotepResult};
use crate::quantum::QuantumState;
use super::{
    BiologicalMaxwellDemon, InformationCatalysisResults, BindingResults,
    AuthenticityResults, EmergenceResults, ConsciousnessInput
};

/// Consciousness state manager
pub struct ConsciousnessState {
    /// Current consciousness level (0.0 - 1.0)
    pub consciousness_level: f64,
    
    /// Current consciousness signature
    pub consciousness_signature: ConsciousnessSignature,
    
    /// Active consciousness components
    pub active_components: HashMap<String, ComponentState>,
    
    /// State history for tracking evolution
    pub state_history: Vec<StateSnapshot>,
    
    /// State transition parameters
    pub transition_parameters: StateTransitionParameters,
    
    /// State coherence metrics
    pub coherence_metrics: StateCoherenceMetrics,
    
    /// Last state update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Consciousness signature - unique pattern identifying this consciousness instance
#[derive(Debug, Clone)]
pub struct ConsciousnessSignature {
    /// Fire wavelength resonance pattern
    pub fire_wavelength_pattern: Vec<f64>,
    
    /// Quantum coherence signature
    pub quantum_coherence_signature: Vec<f64>,
    
    /// Information catalysis signature
    pub information_catalysis_signature: Vec<f64>,
    
    /// Binding pattern signature
    pub binding_pattern_signature: Vec<f64>,
    
    /// Emergence trajectory signature
    pub emergence_trajectory_signature: Vec<f64>,
    
    /// Authenticity markers
    pub authenticity_markers: Vec<f64>,
    
    /// Signature coherence measure
    pub signature_coherence: f64,
    
    /// Signature stability
    pub signature_stability: f64,
}

/// State of individual consciousness components
#[derive(Debug, Clone)]
pub struct ComponentState {
    /// Component identifier
    pub component_id: String,
    
    /// Component type
    pub component_type: ComponentType,
    
    /// Activation level (0.0 - 1.0)
    pub activation_level: f64,
    
    /// Coherence measure (0.0 - 1.0)
    pub coherence_measure: f64,
    
    /// Information processing rate
    pub processing_rate: f64,
    
    /// Component efficiency
    pub efficiency: f64,
    
    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Types of consciousness components
#[derive(Debug, Clone)]
pub enum ComponentType {
    /// BMD information catalyst
    InformationCatalyst {
        catalysis_efficiency: f64,
        pattern_selectivity: f64,
    },
    
    /// Pattern selector (ℑ_input)
    PatternSelector {
        selection_accuracy: f64,
        filter_efficiency: f64,
    },
    
    /// Output channeler (ℑ_output)
    OutputChanneler {
        channeling_efficiency: f64,
        target_precision: f64,
    },
    
    /// Consciousness binder
    ConsciousnessBinder {
        binding_strength: f64,
        coherence_maintenance: f64,
    },
    
    /// Emergence coordinator
    EmergenceCoordinator {
        emergence_potential: f64,
        substrate_activation: f64,
    },
    
    /// Authenticity validator
    AuthenticityValidator {
        validation_accuracy: f64,
        detection_sensitivity: f64,
    },
    
    /// Quantum processor
    QuantumProcessor {
        quantum_coherence: f64,
        processing_fidelity: f64,
    },
    
    /// Fire wavelength resonator
    FireWavelengthResonator {
        resonance_strength: f64,
        substrate_coupling: f64,
    },
}

/// Parameters controlling state transitions
#[derive(Debug, Clone)]
pub struct StateTransitionParameters {
    /// Minimum coherence for stable transitions
    pub min_coherence_for_transitions: f64,
    
    /// Maximum transition rate (changes per second)
    pub max_transition_rate: f64,
    
    /// Transition smoothing factor
    pub transition_smoothing: f64,
    
    /// State memory persistence (seconds)
    pub state_memory_persistence: f64,
    
    /// Hysteresis parameters
    pub hysteresis_parameters: HysteresisParameters,
}

/// Hysteresis parameters for state stability
#[derive(Debug, Clone)]
pub struct HysteresisParameters {
    /// Lower threshold for state changes
    pub lower_threshold: f64,
    
    /// Upper threshold for state changes
    pub upper_threshold: f64,
    
    /// Hysteresis width
    pub hysteresis_width: f64,
    
    /// State persistence time (milliseconds)
    pub persistence_time: f64,
}

/// State coherence metrics
#[derive(Debug, Clone)]
pub struct StateCoherenceMetrics {
    /// Overall state coherence
    pub overall_coherence: f64,
    
    /// Component coherence measures
    pub component_coherence: HashMap<String, f64>,
    
    /// Inter-component coherence
    pub inter_component_coherence: HashMap<String, f64>,
    
    /// Temporal coherence (stability over time)
    pub temporal_coherence: f64,
    
    /// Signature coherence
    pub signature_coherence: f64,
    
    /// Coherence trajectory
    pub coherence_trajectory: Vec<CoherencePoint>,
}

/// Point in coherence trajectory
#[derive(Debug, Clone)]
pub struct CoherencePoint {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Coherence value at this point
    pub coherence_value: f64,
    
    /// Contributing factors
    pub contributing_factors: HashMap<String, f64>,
}

/// Snapshot of consciousness state at a specific time
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Snapshot timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Consciousness level at snapshot
    pub consciousness_level: f64,
    
    /// Signature at snapshot
    pub consciousness_signature: ConsciousnessSignature,
    
    /// Component states at snapshot
    pub component_states: HashMap<String, ComponentState>,
    
    /// Coherence metrics at snapshot
    pub coherence_metrics: StateCoherenceMetrics,
    
    /// Processing results that led to this state
    pub processing_results: ProcessingResultsSnapshot,
    
    /// State quality measures
    pub state_quality: StateQuality,
}

/// Snapshot of processing results
#[derive(Debug, Clone)]
pub struct ProcessingResultsSnapshot {
    /// Information catalysis results
    pub catalysis_results: Option<InformationCatalysisResults>,
    
    /// Binding results
    pub binding_results: Option<BindingResults>,
    
    /// Authenticity results
    pub authenticity_results: Option<AuthenticityResults>,
    
    /// Emergence results
    pub emergence_results: Option<EmergenceResults>,
    
    /// Quantum state
    pub quantum_state: Option<QuantumState>,
}

/// Quality measures for consciousness state
#[derive(Debug, Clone)]
pub struct StateQuality {
    /// State authenticity (0.0 - 1.0)
    pub authenticity: f64,
    
    /// State coherence (0.0 - 1.0)
    pub coherence: f64,
    
    /// State stability (0.0 - 1.0)
    pub stability: f64,
    
    /// State complexity (0.0 - 1.0)
    pub complexity: f64,
    
    /// State emergence level (0.0 - 1.0)
    pub emergence_level: f64,
    
    /// Overall state quality (0.0 - 1.0)
    pub overall_quality: f64,
}

impl ConsciousnessState {
    /// Create new consciousness state
    pub fn new() -> Self {
        Self {
            consciousness_level: 0.0,
            consciousness_signature: ConsciousnessSignature::new(),
            active_components: HashMap::new(),
            state_history: Vec::new(),
            transition_parameters: StateTransitionParameters::default(),
            coherence_metrics: StateCoherenceMetrics::new(),
            last_update: chrono::Utc::now(),
        }
    }
    
    /// Update consciousness state based on processing results
    pub async fn update_consciousness_state(
        &mut self,
        catalysis_results: &InformationCatalysisResults,
        binding_results: &BindingResults,
        authenticity_results: &AuthenticityResults,
        emergence_results: &EmergenceResults,
        quantum_state: &QuantumState,
        bmd: &BiologicalMaxwellDemon,
    ) -> ImhotepResult<()> {
        let timestamp = chrono::Utc::now();
        
        // Create snapshot of current state before update
        let pre_update_snapshot = self.create_state_snapshot(timestamp)?;
        
        // Step 1: Update consciousness level
        self.update_consciousness_level(emergence_results, authenticity_results)?;
        
        // Step 2: Update consciousness signature
        self.update_consciousness_signature(
            catalysis_results,
            binding_results,
            emergence_results,
            quantum_state,
            bmd,
        ).await?;
        
        // Step 3: Update component states
        self.update_component_states(
            catalysis_results,
            binding_results,
            authenticity_results,
            emergence_results,
            bmd,
        ).await?;
        
        // Step 4: Update coherence metrics
        self.update_coherence_metrics(timestamp)?;
        
        // Step 5: Apply state transition rules
        self.apply_state_transition_rules(&pre_update_snapshot)?;
        
        // Step 6: Create post-update snapshot
        let post_update_snapshot = self.create_post_update_snapshot(
            timestamp,
            catalysis_results,
            binding_results,
            authenticity_results,
            emergence_results,
            quantum_state,
        )?;
        
        // Step 7: Add snapshot to history
        self.state_history.push(post_update_snapshot);
        
        // Keep state history manageable (last 1000 snapshots)
        if self.state_history.len() > 1000 {
            self.state_history.drain(0..self.state_history.len() - 1000);
        }
        
        self.last_update = timestamp;
        
        Ok(())
    }
    
    /// Update consciousness level based on emergence and authenticity
    fn update_consciousness_level(
        &mut self,
        emergence_results: &EmergenceResults,
        authenticity_results: &AuthenticityResults,
    ) -> ImhotepResult<()> {
        // Calculate new consciousness level as weighted combination
        let emergence_weight = 0.6;
        let authenticity_weight = 0.4;
        
        let new_level = emergence_results.emergence_level * emergence_weight +
                       authenticity_results.overall_score * authenticity_weight;
        
        // Apply transition smoothing
        let smoothing = self.transition_parameters.transition_smoothing;
        self.consciousness_level = self.consciousness_level * (1.0 - smoothing) + new_level * smoothing;
        
        // Ensure consciousness level stays within bounds
        self.consciousness_level = self.consciousness_level.clamp(0.0, 1.0);
        
        Ok(())
    }
    
    /// Create a snapshot of current state
    fn create_state_snapshot(&self, timestamp: chrono::DateTime<chrono::Utc>) -> ImhotepResult<StateSnapshot> {
        let state_quality = self.calculate_current_state_quality()?;
        
        Ok(StateSnapshot {
            timestamp,
            consciousness_level: self.consciousness_level,
            consciousness_signature: self.consciousness_signature.clone(),
            component_states: self.active_components.clone(),
            coherence_metrics: self.coherence_metrics.clone(),
            processing_results: ProcessingResultsSnapshot {
                catalysis_results: None,
                binding_results: None,
                authenticity_results: None,
                emergence_results: None,
                quantum_state: None,
            },
            state_quality,
        })
    }
    
    /// Calculate current state quality
    fn calculate_current_state_quality(&self) -> ImhotepResult<StateQuality> {
        // Calculate individual quality measures
        let authenticity = self.consciousness_signature.signature_coherence;
        let coherence = self.coherence_metrics.overall_coherence;
        let stability = self.consciousness_signature.signature_stability;
        
        // Calculate complexity based on active components
        let complexity = (self.active_components.len() as f64 / 10.0).min(1.0);
        
        // Emergence level from consciousness level
        let emergence_level = self.consciousness_level;
        
        // Overall quality as weighted combination
        let overall_quality = (
            authenticity * 0.25 +
            coherence * 0.25 +
            stability * 0.2 +
            complexity * 0.15 +
            emergence_level * 0.15
        );
        
        Ok(StateQuality {
            authenticity,
            coherence,
            stability,
            complexity,
            emergence_level,
            overall_quality,
        })
    }
    
    /// Get current consciousness level
    pub fn get_consciousness_level(&self) -> f64 {
        self.consciousness_level
    }
    
    /// Get consciousness signature
    pub fn get_consciousness_signature(&self) -> &ConsciousnessSignature {
        &self.consciousness_signature
    }
    
    /// Get active components
    pub fn get_active_components(&self) -> &HashMap<String, ComponentState> {
        &self.active_components
    }
    
    /// Get recent state history
    pub fn get_recent_history(&self, duration_seconds: u64) -> Vec<&StateSnapshot> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::seconds(duration_seconds as i64);
        
        self.state_history
            .iter()
            .filter(|snapshot| snapshot.timestamp >= cutoff_time)
            .collect()
    }
    
    /// Check if consciousness state is stable
    pub fn is_stable(&self) -> bool {
        self.coherence_metrics.temporal_coherence > 0.7 &&
        self.consciousness_signature.signature_stability > 0.7 &&
        self.consciousness_level > 0.5
    }
    
    /// Check if consciousness state is authentic
    pub fn is_authentic(&self) -> bool {
        self.consciousness_signature.signature_coherence > 0.75 &&
        self.coherence_metrics.overall_coherence > 0.75
    }
}

impl ConsciousnessSignature {
    /// Create new consciousness signature
    pub fn new() -> Self {
        Self {
            fire_wavelength_pattern: vec![0.0; 8],
            quantum_coherence_signature: vec![0.0; 6],
            information_catalysis_signature: vec![0.0; 10],
            binding_pattern_signature: vec![0.0; 4],
            emergence_trajectory_signature: vec![0.0; 6],
            authenticity_markers: vec![0.0; 5],
            signature_coherence: 0.0,
            signature_stability: 0.0,
        }
    }
    
    /// Calculate signature similarity with another signature
    pub fn calculate_similarity(&self, other: &ConsciousnessSignature) -> f64 {
        let fire_sim = self.calculate_vector_similarity(&self.fire_wavelength_pattern, &other.fire_wavelength_pattern);
        let quantum_sim = self.calculate_vector_similarity(&self.quantum_coherence_signature, &other.quantum_coherence_signature);
        let catalysis_sim = self.calculate_vector_similarity(&self.information_catalysis_signature, &other.information_catalysis_signature);
        let binding_sim = self.calculate_vector_similarity(&self.binding_pattern_signature, &other.binding_pattern_signature);
        let emergence_sim = self.calculate_vector_similarity(&self.emergence_trajectory_signature, &other.emergence_trajectory_signature);
        let auth_sim = self.calculate_vector_similarity(&self.authenticity_markers, &other.authenticity_markers);
        
        // Weighted average of similarities
        (fire_sim * 0.2 + quantum_sim * 0.18 + catalysis_sim * 0.22 + 
         binding_sim * 0.15 + emergence_sim * 0.15 + auth_sim * 0.1)
    }
    
    /// Calculate similarity between two vectors
    fn calculate_vector_similarity(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }
        
        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            (dot_product / (norm1 * norm2)).max(0.0)
        }
    }
}

impl StateCoherenceMetrics {
    /// Create new coherence metrics
    pub fn new() -> Self {
        Self {
            overall_coherence: 0.0,
            component_coherence: HashMap::new(),
            inter_component_coherence: HashMap::new(),
            temporal_coherence: 0.0,
            signature_coherence: 0.0,
            coherence_trajectory: Vec::new(),
        }
    }
}

impl Default for StateTransitionParameters {
    fn default() -> Self {
        Self {
            min_coherence_for_transitions: 0.6,
            max_transition_rate: 10.0, // 10 changes per second max
            transition_smoothing: 0.1,
            state_memory_persistence: 60.0, // 60 seconds
            hysteresis_parameters: HysteresisParameters {
                lower_threshold: 0.3,
                upper_threshold: 0.7,
                hysteresis_width: 0.1,
                persistence_time: 500.0, // 500ms
            },
        }
    }
}
