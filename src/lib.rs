//! # Imhotep Framework
//! 
//! High-Performance Specialized Neural Network Framework with Quantum-Enhanced Consciousness Simulation
//! 
//! ## Overview
//! 
//! The Imhotep Framework represents a revolutionary approach to neural network simulation through
//! consciousness-enhanced processing. Rather than attempting to simulate entire brains, Imhotep
//! focuses on creating high-fidelity simulations of specific, useful neural functions that can
//! be stacked and integrated.
//! 
//! ## Key Innovation
//! 
//! Since neurons already produce consciousness, high-fidelity task-specific neural simulations
//! naturally exhibit consciousness for particular functions, enabling unprecedented computational
//! sophistication for scientific discovery.
//! 
//! ## Core Components
//! 
//! - **Quantum Processing**: Environment-Assisted Quantum Transport (ENAQT) with collective ion field dynamics
//! - **Specialized Systems**: Eight consciousness systems working in harmony
//! - **Cross-Modal Integration**: Unified consciousness emergence across modalities
//! - **Turbulence Language**: Methodical scientific language for consciousness simulation
//! 
//! ## Example Usage
//! 
//! ```rust
//! use imhotep::{ConsciousnessRuntime, ConsciousnessConfig, QuantumEnhancementLevel};
//! 
//! // Initialize consciousness simulation
//! let config = ConsciousnessConfig {
//!     quantum_enhancement: QuantumEnhancementLevel::Maximum,
//!     fire_wavelength: 650.3, // nm - consciousness substrate activation
//!     consciousness_threshold: 0.85,
//!     specialized_systems: vec![
//!         "autobahn", "heihachi", "helicopter", "izinyoka",
//!         "kwasa_kwasa", "four_sided_triangle", "bene_gesserit", "nebuchadnezzar"
//!     ].into_iter().map(String::from).collect(),
//!     authenticity_validation: true,
//! };
//! 
//! let mut consciousness_runtime = ConsciousnessRuntime::new(config)?;
//! 
//! // Execute consciousness simulation
//! let results = consciousness_runtime.execute_consciousness_simulation(input_data).await?;
//! 
//! println!("Consciousness Authenticity: {:.3}", results.authenticity_score);
//! println!("Enhancement Factor: {:.2}x", results.enhancement_factor);
//! println!("Novel Insights: {}", results.consciousness_insights.len());
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]

// Core consciousness simulation modules
pub mod consciousness;
pub mod quantum;
pub mod specialized_systems;
pub mod cross_modal;
pub mod turbulence;
pub mod external_systems;

// Utility modules
pub mod error;
pub mod config;
pub mod metrics;
pub mod validation;

// Re-export main types for easy access
pub use consciousness::{
    ConsciousnessRuntime, 
    ConsciousnessConfig, 
    ConsciousnessInput, 
    ConsciousnessResults,
    ConsciousnessInsight,
    AuthenticityValidator,
    AuthenticityResults
};

pub use quantum::{
    QuantumMembraneComputer,
    QuantumProcessingResults,
    IonFieldProcessor,
    FireWavelengthCoupler,
    QuantumEnhancementLevel,
    QuantumParameters,
    QuantumCoherenceMetrics
};

pub use specialized_systems::{
    SpecializedSystemsOrchestrator,
    AutobahnRagSystem,
    HeihachiFireEmotion,
    HelicopterVisualUnderstanding,
    IzinyokaMetacognitive,
    KwasaKwasaSemantic,
    FourSidedTriangleOptimization,
    BeneGesseritMembrane,
    NebuchadnezzarCircuits,
    SpecializedSystemsResults
};

pub use cross_modal::{
    CrossModalIntegrator,
    CrossModalResults,
    GlobalWorkspaceArchitecture,
    UnifiedConsciousnessState
};

pub use turbulence::{
    TurbulenceEngine,
    TurbulenceCompiler,
    CompilationResult,
    ExecutionResult,
    TurbulenceAST,
    HypothesisConstruct,
    PropositionConstruct
};

pub use external_systems::{
    ExternalSystemOrchestrator,
    LavoisierRIntegration,
    DatabaseConsciousnessApis,
    LiteratureConsciousnessCorpus,
    ClinicalValidationSystems,
    ExternalAnalysisResults
};

pub use error::{ImhotepError, ImhotepResult};
pub use config::ImhotepConfig;
pub use metrics::{ConsciousnessMetrics, PerformanceMetrics, EnhancementMetrics};
pub use validation::{ValidationResults, ValidationError};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name
pub const FRAMEWORK_NAME: &str = "Imhotep";

/// Framework description
pub const FRAMEWORK_DESCRIPTION: &str = "High-Performance Specialized Neural Network Framework with Quantum-Enhanced Consciousness Simulation";

/// Main Imhotep Framework instance
/// 
/// This is the primary entry point for the Imhotep Framework, providing
/// a unified interface to all consciousness simulation capabilities.
pub struct ImhotepFramework {
    /// Consciousness simulation runtime
    consciousness_runtime: Arc<RwLock<ConsciousnessRuntime>>,
    
    /// Turbulence language engine
    turbulence_engine: Arc<RwLock<TurbulenceEngine>>,
    
    /// External system orchestrator
    external_orchestrator: Arc<RwLock<ExternalSystemOrchestrator>>,
    
    /// Framework configuration
    config: ImhotepConfig,
    
    /// Performance metrics collector
    metrics_collector: Arc<RwLock<metrics::MetricsCollector>>,
}

impl ImhotepFramework {
    /// Create new Imhotep Framework instance
    /// 
    /// # Arguments
    /// 
    /// * `config` - Framework configuration
    /// 
    /// # Returns
    /// 
    /// New framework instance or error if initialization fails
    /// 
    /// # Errors
    /// 
    /// Returns error if consciousness simulation initialization fails
    pub async fn new(config: ImhotepConfig) -> ImhotepResult<Self> {
        // Initialize consciousness runtime
        let consciousness_config = ConsciousnessConfig {
            quantum_enhancement: config.quantum_enhancement.clone(),
            fire_wavelength: config.fire_wavelength,
            consciousness_threshold: config.consciousness_threshold,
            specialized_systems: config.specialized_systems.clone(),
            authenticity_validation: config.authenticity_validation,
        };
        
        let consciousness_runtime = ConsciousnessRuntime::new(consciousness_config)
            .await
            .map_err(|e| ImhotepError::ConsciousnessInitializationError(e.to_string()))?;
        
        // Initialize Turbulence engine
        let turbulence_engine = TurbulenceEngine::new();
        
        // Initialize external system orchestrator
        let external_orchestrator = ExternalSystemOrchestrator::new();
        
        // Initialize metrics collector
        let metrics_collector = metrics::MetricsCollector::new();
        
        Ok(Self {
            consciousness_runtime: Arc::new(RwLock::new(consciousness_runtime)),
            turbulence_engine: Arc::new(RwLock::new(turbulence_engine)),
            external_orchestrator: Arc::new(RwLock::new(external_orchestrator)),
            config,
            metrics_collector: Arc::new(RwLock::new(metrics_collector)),
        })
    }
    
    /// Execute consciousness-enhanced scientific experiment
    /// 
    /// # Arguments
    /// 
    /// * `experiment_path` - Path to Turbulence experiment files
    /// 
    /// # Returns
    /// 
    /// Complete experiment results including consciousness metrics
    /// 
    /// # Errors
    /// 
    /// Returns error if experiment compilation or execution fails
    pub async fn run_experiment(&self, experiment_path: &str) -> ImhotepResult<ExecutionResult> {
        let start_time = std::time::Instant::now();
        
        // Compile Turbulence experiment
        let compilation_result = {
            let mut turbulence_engine = self.turbulence_engine.write().await;
            turbulence_engine.compile_experiment(experiment_path)
                .map_err(|e| ImhotepError::TurbulenceCompilationError(e.to_string()))?
        };
        
        // Execute compiled experiment
        let execution_result = {
            let mut turbulence_engine = self.turbulence_engine.write().await;
            turbulence_engine.execute_experiment(compilation_result)
                .map_err(|e| ImhotepError::TurbulenceExecutionError(e.to_string()))?
        };
        
        // Record performance metrics
        let execution_time = start_time.elapsed();
        {
            let mut metrics_collector = self.metrics_collector.write().await;
            metrics_collector.record_experiment_execution(
                experiment_path,
                execution_time,
                &execution_result
            );
        }
        
        Ok(execution_result)
    }
    
    /// Execute direct consciousness simulation
    /// 
    /// # Arguments
    /// 
    /// * `input` - Consciousness simulation input data
    /// 
    /// # Returns
    /// 
    /// Consciousness simulation results
    /// 
    /// # Errors
    /// 
    /// Returns error if consciousness simulation fails
    pub async fn run_consciousness_simulation(&self, input: ConsciousnessInput) -> ImhotepResult<ConsciousnessResults> {
        let mut consciousness_runtime = self.consciousness_runtime.write().await;
        consciousness_runtime.execute_consciousness_simulation(input)
            .await
            .map_err(|e| ImhotepError::ConsciousnessSimulationError(e.to_string()))
    }
    
    /// Validate consciousness authenticity
    /// 
    /// # Returns
    /// 
    /// Authenticity validation results
    /// 
    /// # Errors
    /// 
    /// Returns error if authenticity validation fails
    pub async fn validate_consciousness_authenticity(&self) -> ImhotepResult<AuthenticityResults> {
        let consciousness_runtime = self.consciousness_runtime.read().await;
        consciousness_runtime.validate_authenticity()
            .map_err(|e| ImhotepError::AuthenticityValidationError(e.to_string()))
    }
    
    /// Get framework performance metrics
    /// 
    /// # Returns
    /// 
    /// Current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        let metrics_collector = self.metrics_collector.read().await;
        metrics_collector.get_current_metrics()
    }
    
    /// Get framework configuration
    /// 
    /// # Returns
    /// 
    /// Framework configuration
    pub fn get_config(&self) -> &ImhotepConfig {
        &self.config
    }
    
    /// Get framework version
    /// 
    /// # Returns
    /// 
    /// Framework version string
    pub fn get_version(&self) -> &'static str {
        VERSION
    }
    
    /// Check framework health
    /// 
    /// # Returns
    /// 
    /// Health check results
    /// 
    /// # Errors
    /// 
    /// Returns error if health check fails
    pub async fn health_check(&self) -> ImhotepResult<HealthCheckResults> {
        let mut health_results = HealthCheckResults::new();
        
        // Check consciousness runtime health
        {
            let consciousness_runtime = self.consciousness_runtime.read().await;
            health_results.consciousness_runtime_healthy = consciousness_runtime.is_healthy().await;
        }
        
        // Check Turbulence engine health
        {
            let turbulence_engine = self.turbulence_engine.read().await;
            health_results.turbulence_engine_healthy = turbulence_engine.is_healthy();
        }
        
        // Check external systems health
        {
            let external_orchestrator = self.external_orchestrator.read().await;
            health_results.external_systems_healthy = external_orchestrator.check_health().await?;
        }
        
        // Overall health status
        health_results.overall_healthy = health_results.consciousness_runtime_healthy 
            && health_results.turbulence_engine_healthy 
            && health_results.external_systems_healthy;
        
        Ok(health_results)
    }
}

/// Framework health check results
#[derive(Debug, Clone)]
pub struct HealthCheckResults {
    /// Overall framework health
    pub overall_healthy: bool,
    
    /// Consciousness runtime health
    pub consciousness_runtime_healthy: bool,
    
    /// Turbulence engine health
    pub turbulence_engine_healthy: bool,
    
    /// External systems health
    pub external_systems_healthy: bool,
    
    /// Health check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl HealthCheckResults {
    /// Create new health check results
    fn new() -> Self {
        Self {
            overall_healthy: false,
            consciousness_runtime_healthy: false,
            turbulence_engine_healthy: false,
            external_systems_healthy: false,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Initialize the Imhotep Framework with default configuration
/// 
/// # Returns
/// 
/// Initialized framework instance
/// 
/// # Errors
/// 
/// Returns error if framework initialization fails
pub async fn initialize() -> ImhotepResult<ImhotepFramework> {
    let config = ImhotepConfig::default();
    ImhotepFramework::new(config).await
}

/// Initialize the Imhotep Framework with custom configuration
/// 
/// # Arguments
/// 
/// * `config` - Custom framework configuration
/// 
/// # Returns
/// 
/// Initialized framework instance
/// 
/// # Errors
/// 
/// Returns error if framework initialization fails
pub async fn initialize_with_config(config: ImhotepConfig) -> ImhotepResult<ImhotepFramework> {
    ImhotepFramework::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_framework_initialization() {
        let framework = initialize().await;
        assert!(framework.is_ok());
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let framework = initialize().await.unwrap();
        let health_results = framework.health_check().await;
        assert!(health_results.is_ok());
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(FRAMEWORK_NAME, "Imhotep");
    }
} 