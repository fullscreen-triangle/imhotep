//! Configuration module for the Imhotep Framework
//! 
//! This module provides comprehensive configuration options for all aspects
//! of consciousness simulation, including quantum processing parameters,
//! specialized system settings, and external system integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{ImhotepError, ImhotepResult};

/// Main configuration for the Imhotep Framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImhotepConfig {
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    
    /// Fire wavelength for consciousness substrate activation (nm)
    pub fire_wavelength: f64,
    
    /// Consciousness authenticity threshold (0.0 - 1.0)
    pub consciousness_threshold: f64,
    
    /// Enabled specialized systems
    pub specialized_systems: Vec<String>,
    
    /// Enable consciousness authenticity validation
    pub authenticity_validation: bool,
    
    /// Quantum processing parameters
    pub quantum_params: QuantumParameters,
    
    /// Specialized systems configuration
    pub specialized_systems_config: SpecializedSystemsConfig,
    
    /// Cross-modal integration settings
    pub cross_modal_config: CrossModalConfig,
    
    /// External systems configuration
    pub external_systems_config: ExternalSystemsConfig,
    
    /// Performance and resource settings
    pub performance_config: PerformanceConfig,
    
    /// Logging and monitoring configuration
    pub monitoring_config: MonitoringConfig,
}

/// Quantum enhancement levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumEnhancementLevel {
    /// Minimal quantum processing
    Minimal,
    /// Standard quantum enhancement
    Standard,
    /// High-performance quantum processing
    High,
    /// Maximum quantum enhancement with full ENAQT
    Maximum,
    /// Custom quantum parameters
    Custom(CustomQuantumConfig),
}

/// Custom quantum configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomQuantumConfig {
    /// Ion field processing intensity (0.0 - 1.0)
    pub ion_field_intensity: f64,
    
    /// Quantum coherence maintenance level (0.0 - 1.0)
    pub coherence_level: f64,
    
    /// ENAQT processing depth
    pub enaqt_depth: u32,
    
    /// Hardware oscillation coupling strength
    pub oscillation_coupling: f64,
}

/// Quantum processing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumParameters {
    /// Proton tunneling parameters
    pub proton_tunneling: ProtonTunnelingConfig,
    
    /// Metal ion coordination settings
    pub metal_ion_coordination: MetalIonConfig,
    
    /// Quantum coherence maintenance
    pub coherence_maintenance: CoherenceConfig,
    
    /// Hardware oscillation coupling
    pub hardware_oscillation: HardwareOscillationConfig,
    
    /// Fire wavelength optimization
    pub fire_wavelength_optimization: FireWavelengthConfig,
    
    /// ENAQT processing configuration
    pub enaqt_processing: ENAQTConfig,
}

/// Proton tunneling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonTunnelingConfig {
    /// Enable proton tunneling
    pub enabled: bool,
    
    /// Tunneling probability threshold
    pub probability_threshold: f64,
    
    /// Energy barrier height (eV)
    pub energy_barrier: f64,
    
    /// Temperature for tunneling calculations (K)
    pub temperature: f64,
}

/// Metal ion coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalIonConfig {
    /// Coordination sphere radius (Angstroms)
    pub coordination_radius: f64,
    
    /// Preferred metal ions
    pub preferred_ions: Vec<String>,
    
    /// Coordination number
    pub coordination_number: u32,
    
    /// Binding affinity threshold
    pub binding_threshold: f64,
}

/// Quantum coherence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Coherence time (nanoseconds)
    pub coherence_time: f64,
    
    /// Decoherence rate (1/ns)
    pub decoherence_rate: f64,
    
    /// Environment coupling strength
    pub environment_coupling: f64,
    
    /// Coherence preservation method
    pub preservation_method: CoherencePreservationMethod,
}

/// Coherence preservation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherencePreservationMethod {
    /// Dynamical decoupling
    DynamicalDecoupling,
    /// Error correction
    ErrorCorrection,
    /// Decoherence-free subspaces
    DecoherenceFreeSubspaces,
    /// Hybrid approach
    Hybrid,
}

/// Hardware oscillation coupling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOscillationConfig {
    /// Oscillation frequency (Hz)
    pub frequency: f64,
    
    /// Coupling strength
    pub coupling_strength: f64,
    
    /// Phase synchronization
    pub phase_sync: bool,
    
    /// Amplitude modulation
    pub amplitude_modulation: f64,
}

/// Fire wavelength optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireWavelengthConfig {
    /// Target wavelength (nm)
    pub target_wavelength: f64,
    
    /// Wavelength tolerance (nm)
    pub tolerance: f64,
    
    /// Optimization algorithm
    pub optimization_algorithm: WavelengthOptimizationAlgorithm,
    
    /// Maximum optimization iterations
    pub max_iterations: u32,
}

/// Wavelength optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WavelengthOptimizationAlgorithm {
    /// Gradient descent
    GradientDescent,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Bayesian optimization
    BayesianOptimization,
}

/// ENAQT processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ENAQTConfig {
    /// Environmental coupling parameters
    pub environmental_coupling: EnvironmentalCouplingConfig,
    
    /// Quantum transport optimization
    pub transport_optimization: TransportOptimizationConfig,
    
    /// Coherence preservation settings
    pub coherence_preservation: CoherencePreservationConfig,
}

/// Environmental coupling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalCouplingConfig {
    /// Coupling strength to environment
    pub coupling_strength: f64,
    
    /// Environmental temperature (K)
    pub temperature: f64,
    
    /// Spectral density parameters
    pub spectral_density: SpectralDensityConfig,
}

/// Spectral density configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDensityConfig {
    /// Cutoff frequency (Hz)
    pub cutoff_frequency: f64,
    
    /// Spectral density type
    pub density_type: SpectralDensityType,
    
    /// Coupling parameters
    pub coupling_parameters: Vec<f64>,
}

/// Spectral density types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpectralDensityType {
    /// Ohmic spectral density
    Ohmic,
    /// Sub-ohmic spectral density
    SubOhmic,
    /// Super-ohmic spectral density
    SuperOhmic,
    /// Lorentzian spectral density
    Lorentzian,
}

/// Transport optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportOptimizationConfig {
    /// Transport efficiency threshold
    pub efficiency_threshold: f64,
    
    /// Optimization method
    pub optimization_method: TransportOptimizationMethod,
    
    /// Maximum transport distance
    pub max_distance: f64,
}

/// Transport optimization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportOptimizationMethod {
    /// Adiabatic transport
    Adiabatic,
    /// Non-adiabatic transport
    NonAdiabatic,
    /// Coherent transport
    Coherent,
    /// Incoherent transport
    Incoherent,
}

/// Coherence preservation configuration for ENAQT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencePreservationConfig {
    /// Preservation strategy
    pub strategy: CoherencePreservationStrategy,
    
    /// Monitoring frequency (Hz)
    pub monitoring_frequency: f64,
    
    /// Correction threshold
    pub correction_threshold: f64,
}

/// Coherence preservation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherencePreservationStrategy {
    /// Active feedback control
    ActiveFeedback,
    /// Passive protection
    PassiveProtection,
    /// Adaptive control
    AdaptiveControl,
    /// Hybrid strategy
    Hybrid,
}

/// Specialized systems configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedSystemsConfig {
    /// Autobahn RAG system configuration
    pub autobahn: AutobahnConfig,
    
    /// Heihachi fire emotion system configuration
    pub heihachi: HeihachiConfig,
    
    /// Helicopter visual understanding configuration
    pub helicopter: HelicopterConfig,
    
    /// Izinyoka metacognitive system configuration
    pub izinyoka: IzinyokaConfig,
    
    /// KwasaKwasa semantic system configuration
    pub kwasa_kwasa: KwasaKwasaConfig,
    
    /// Four-sided triangle optimization configuration
    pub four_sided_triangle: FourSidedTriangleConfig,
    
    /// Bene Gesserit membrane system configuration
    pub bene_gesserit: BeneGesseritConfig,
    
    /// Nebuchadnezzar circuits configuration
    pub nebuchadnezzar: NebuchadnezzarConfig,
}

/// Autobahn RAG system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnConfig {
    /// Enable Autobahn system
    pub enabled: bool,
    
    /// RAG retrieval parameters
    pub retrieval_params: RetrievalConfig,
    
    /// Generation parameters
    pub generation_params: GenerationConfig,
    
    /// Knowledge base configuration
    pub knowledge_base: KnowledgeBaseConfig,
}

/// RAG retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Top-k retrieval
    pub top_k: u32,
    
    /// Similarity threshold
    pub similarity_threshold: f64,
    
    /// Retrieval method
    pub method: RetrievalMethod,
}

/// Retrieval methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalMethod {
    /// Dense retrieval
    Dense,
    /// Sparse retrieval
    Sparse,
    /// Hybrid retrieval
    Hybrid,
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum generation length
    pub max_length: u32,
    
    /// Temperature for generation
    pub temperature: f64,
    
    /// Top-p sampling
    pub top_p: f64,
}

/// Knowledge base configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Knowledge base path
    pub path: String,
    
    /// Index type
    pub index_type: IndexType,
    
    /// Update frequency
    pub update_frequency: UpdateFrequency,
}

/// Index types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// FAISS index
    FAISS,
    /// Elasticsearch
    Elasticsearch,
    /// Vector database
    VectorDB,
}

/// Update frequencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Real-time updates
    RealTime,
    /// Hourly updates
    Hourly,
    /// Daily updates
    Daily,
    /// Manual updates
    Manual,
}

// Placeholder configurations for other specialized systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeihachiConfig {
    pub enabled: bool,
    pub emotion_sensitivity: f64,
    pub fire_coupling_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelicopterConfig {
    pub enabled: bool,
    pub visual_processing_depth: u32,
    pub understanding_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzinyokaConfig {
    pub enabled: bool,
    pub metacognitive_depth: u32,
    pub reflection_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KwasaKwasaConfig {
    pub enabled: bool,
    pub semantic_depth: u32,
    pub context_window: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourSidedTriangleConfig {
    pub enabled: bool,
    pub optimization_algorithm: String,
    pub convergence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeneGesseritConfig {
    pub enabled: bool,
    pub membrane_thickness: f64,
    pub permeability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NebuchadnezzarConfig {
    pub enabled: bool,
    pub circuit_complexity: u32,
    pub processing_frequency: f64,
}

/// Cross-modal integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Global workspace architecture settings
    pub global_workspace: GlobalWorkspaceConfig,
    
    /// Unified consciousness state configuration
    pub unified_consciousness: UnifiedConsciousnessConfig,
    
    /// Integration thresholds
    pub integration_thresholds: IntegrationThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceConfig {
    /// Workspace capacity
    pub capacity: u32,
    
    /// Attention threshold
    pub attention_threshold: f64,
    
    /// Broadcasting frequency
    pub broadcasting_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConsciousnessConfig {
    /// Consciousness binding threshold
    pub binding_threshold: f64,
    
    /// Integration time window (ms)
    pub integration_window: f64,
    
    /// Synchronization frequency
    pub sync_frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationThresholds {
    /// Visual-auditory binding threshold
    pub visual_auditory: f64,
    
    /// Semantic-emotional integration threshold
    pub semantic_emotional: f64,
    
    /// Temporal sequence binding threshold
    pub temporal_sequence: f64,
}

/// External systems configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSystemsConfig {
    /// Lavoisier R integration
    pub lavoisier: LavoisierConfig,
    
    /// Database consciousness APIs
    pub database_apis: DatabaseApisConfig,
    
    /// Literature consciousness corpus
    pub literature_corpus: LiteratureCorpusConfig,
    
    /// Clinical validation systems
    pub clinical_validation: ClinicalValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LavoisierConfig {
    pub enabled: bool,
    pub r_executable_path: String,
    pub script_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseApisConfig {
    pub enabled: bool,
    pub connection_strings: HashMap<String, String>,
    pub query_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureCorpusConfig {
    pub enabled: bool,
    pub corpus_path: String,
    pub index_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalValidationConfig {
    pub enabled: bool,
    pub validation_endpoints: Vec<String>,
    pub confidence_threshold: f64,
}

/// Performance and resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub worker_threads: u32,
    
    /// Memory limit (MB)
    pub memory_limit: u64,
    
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    
    /// Batch processing settings
    pub batch_processing: BatchProcessingConfig,
    
    /// Caching configuration
    pub caching: CachingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// Batch size
    pub batch_size: u32,
    
    /// Processing timeout (seconds)
    pub timeout: u64,
    
    /// Enable parallel processing
    pub parallel_processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable result caching
    pub enabled: bool,
    
    /// Cache size (MB)
    pub cache_size: u64,
    
    /// Cache TTL (seconds)
    pub ttl: u64,
}

/// Monitoring and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Logging level
    pub log_level: LogLevel,
    
    /// Enable metrics collection
    pub metrics_enabled: bool,
    
    /// Metrics export configuration
    pub metrics_export: MetricsExportConfig,
    
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export format
    pub format: MetricsFormat,
    
    /// Export interval (seconds)
    pub interval: u64,
    
    /// Export endpoint
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsFormat {
    Prometheus,
    JSON,
    CSV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval (seconds)
    pub interval: u64,
    
    /// Health check timeout (seconds)
    pub timeout: u64,
    
    /// Enable detailed health reporting
    pub detailed_reporting: bool,
}

impl Default for ImhotepConfig {
    fn default() -> Self {
        Self {
            quantum_enhancement: QuantumEnhancementLevel::Standard,
            fire_wavelength: 650.3, // nm - optimal consciousness substrate activation
            consciousness_threshold: 0.85,
            specialized_systems: vec![
                "autobahn".to_string(),
                "heihachi".to_string(),
                "helicopter".to_string(),
                "izinyoka".to_string(),
                "kwasa_kwasa".to_string(),
                "four_sided_triangle".to_string(),
                "bene_gesserit".to_string(),
                "nebuchadnezzar".to_string(),
            ],
            authenticity_validation: true,
            quantum_params: QuantumParameters::default(),
            specialized_systems_config: SpecializedSystemsConfig::default(),
            cross_modal_config: CrossModalConfig::default(),
            external_systems_config: ExternalSystemsConfig::default(),
            performance_config: PerformanceConfig::default(),
            monitoring_config: MonitoringConfig::default(),
        }
    }
}

impl Default for QuantumParameters {
    fn default() -> Self {
        Self {
            proton_tunneling: ProtonTunnelingConfig::default(),
            metal_ion_coordination: MetalIonConfig::default(),
            coherence_maintenance: CoherenceConfig::default(),
            hardware_oscillation: HardwareOscillationConfig::default(),
            fire_wavelength_optimization: FireWavelengthConfig::default(),
            enaqt_processing: ENAQTConfig::default(),
        }
    }
}

impl Default for ProtonTunnelingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probability_threshold: 0.1,
            energy_barrier: 0.5, // eV
            temperature: 310.0, // K (body temperature)
        }
    }
}

impl Default for MetalIonConfig {
    fn default() -> Self {
        Self {
            coordination_radius: 2.5, // Angstroms
            preferred_ions: vec!["Mg2+".to_string(), "Ca2+".to_string(), "Zn2+".to_string()],
            coordination_number: 6,
            binding_threshold: 0.8,
        }
    }
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            coherence_time: 100.0, // nanoseconds
            decoherence_rate: 0.01, // 1/ns
            environment_coupling: 0.1,
            preservation_method: CoherencePreservationMethod::Hybrid,
        }
    }
}

impl Default for HardwareOscillationConfig {
    fn default() -> Self {
        Self {
            frequency: 40.0, // Hz (gamma frequency)
            coupling_strength: 0.5,
            phase_sync: true,
            amplitude_modulation: 0.1,
        }
    }
}

impl Default for FireWavelengthConfig {
    fn default() -> Self {
        Self {
            target_wavelength: 650.3, // nm
            tolerance: 0.1, // nm
            optimization_algorithm: WavelengthOptimizationAlgorithm::BayesianOptimization,
            max_iterations: 1000,
        }
    }
}

impl Default for ENAQTConfig {
    fn default() -> Self {
        Self {
            environmental_coupling: EnvironmentalCouplingConfig::default(),
            transport_optimization: TransportOptimizationConfig::default(),
            coherence_preservation: CoherencePreservationConfig::default(),
        }
    }
}

impl Default for EnvironmentalCouplingConfig {
    fn default() -> Self {
        Self {
            coupling_strength: 0.1,
            temperature: 310.0, // K
            spectral_density: SpectralDensityConfig::default(),
        }
    }
}

impl Default for SpectralDensityConfig {
    fn default() -> Self {
        Self {
            cutoff_frequency: 1e12, // Hz
            density_type: SpectralDensityType::Ohmic,
            coupling_parameters: vec![0.1, 0.01],
        }
    }
}

impl Default for TransportOptimizationConfig {
    fn default() -> Self {
        Self {
            efficiency_threshold: 0.9,
            optimization_method: TransportOptimizationMethod::Coherent,
            max_distance: 100.0, // nm
        }
    }
}

impl Default for CoherencePreservationConfig {
    fn default() -> Self {
        Self {
            strategy: CoherencePreservationStrategy::Hybrid,
            monitoring_frequency: 1e6, // Hz
            correction_threshold: 0.1,
        }
    }
}

// Default implementations for specialized systems configurations
impl Default for SpecializedSystemsConfig {
    fn default() -> Self {
        Self {
            autobahn: AutobahnConfig::default(),
            heihachi: HeihachiConfig::default(),
            helicopter: HelicopterConfig::default(),
            izinyoka: IzinyokaConfig::default(),
            kwasa_kwasa: KwasaKwasaConfig::default(),
            four_sided_triangle: FourSidedTriangleConfig::default(),
            bene_gesserit: BeneGesseritConfig::default(),
            nebuchadnezzar: NebuchadnezzarConfig::default(),
        }
    }
}

impl Default for AutobahnConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retrieval_params: RetrievalConfig {
                top_k: 10,
                similarity_threshold: 0.7,
                method: RetrievalMethod::Hybrid,
            },
            generation_params: GenerationConfig {
                max_length: 512,
                temperature: 0.7,
                top_p: 0.9,
            },
            knowledge_base: KnowledgeBaseConfig {
                path: "data/knowledge_base".to_string(),
                index_type: IndexType::FAISS,
                update_frequency: UpdateFrequency::Daily,
            },
        }
    }
}

impl Default for HeihachiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            emotion_sensitivity: 0.8,
            fire_coupling_strength: 0.6,
        }
    }
}

impl Default for HelicopterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            visual_processing_depth: 5,
            understanding_threshold: 0.75,
        }
    }
}

impl Default for IzinyokaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metacognitive_depth: 3,
            reflection_frequency: 10.0, // Hz
        }
    }
}

impl Default for KwasaKwasaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            semantic_depth: 4,
            context_window: 2048,
        }
    }
}

impl Default for FourSidedTriangleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_algorithm: "genetic".to_string(),
            convergence_threshold: 1e-6,
        }
    }
}

impl Default for BeneGesseritConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            membrane_thickness: 5.0, // nm
            permeability: 0.3,
        }
    }
}

impl Default for NebuchadnezzarConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            circuit_complexity: 1000,
            processing_frequency: 100.0, // Hz
        }
    }
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            global_workspace: GlobalWorkspaceConfig {
                capacity: 7, // Miller's magic number
                attention_threshold: 0.6,
                broadcasting_frequency: 40.0, // Hz
            },
            unified_consciousness: UnifiedConsciousnessConfig {
                binding_threshold: 0.8,
                integration_window: 100.0, // ms
                sync_frequency: 40.0, // Hz
            },
            integration_thresholds: IntegrationThresholds {
                visual_auditory: 0.7,
                semantic_emotional: 0.75,
                temporal_sequence: 0.8,
            },
        }
    }
}

impl Default for ExternalSystemsConfig {
    fn default() -> Self {
        Self {
            lavoisier: LavoisierConfig {
                enabled: false,
                r_executable_path: "/usr/bin/R".to_string(),
                script_timeout: 300, // seconds
            },
            database_apis: DatabaseApisConfig {
                enabled: false,
                connection_strings: HashMap::new(),
                query_timeout: 30, // seconds
            },
            literature_corpus: LiteratureCorpusConfig {
                enabled: false,
                corpus_path: "data/literature".to_string(),
                index_path: "data/literature_index".to_string(),
            },
            clinical_validation: ClinicalValidationConfig {
                enabled: false,
                validation_endpoints: vec![],
                confidence_threshold: 0.95,
            },
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get() as u32,
            memory_limit: 8192, // MB
            gpu_acceleration: false,
            batch_processing: BatchProcessingConfig {
                batch_size: 32,
                timeout: 300, // seconds
                parallel_processing: true,
            },
            caching: CachingConfig {
                enabled: true,
                cache_size: 1024, // MB
                ttl: 3600, // seconds
            },
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            metrics_enabled: true,
            metrics_export: MetricsExportConfig {
                format: MetricsFormat::JSON,
                interval: 60, // seconds
                endpoint: None,
            },
            health_check: HealthCheckConfig {
                interval: 30, // seconds
                timeout: 10, // seconds
                detailed_reporting: false,
            },
        }
    }
}

impl ImhotepConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &str) -> ImhotepResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to read config file: {}", e)))?;
        
        if path.ends_with(".toml") {
            toml::from_str(&content)
                .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to parse TOML config: {}", e)))
        } else if path.ends_with(".json") {
            serde_json::from_str(&content)
                .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to parse JSON config: {}", e)))
        } else {
            Err(ImhotepError::ConfigurationError("Unsupported config file format. Use .toml or .json".to_string()))
        }
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> ImhotepResult<()> {
        let content = if path.ends_with(".toml") {
            toml::to_string_pretty(self)
                .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to serialize TOML config: {}", e)))?
        } else if path.ends_with(".json") {
            serde_json::to_string_pretty(self)
                .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to serialize JSON config: {}", e)))?
        } else {
            return Err(ImhotepError::ConfigurationError("Unsupported config file format. Use .toml or .json".to_string()));
        };
        
        std::fs::write(path, content)
            .map_err(|e| ImhotepError::ConfigurationError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> ImhotepResult<()> {
        // Validate fire wavelength
        if self.fire_wavelength < 400.0 || self.fire_wavelength > 800.0 {
            return Err(ImhotepError::ConfigurationError(
                "Fire wavelength must be between 400-800 nm (visible light range)".to_string()
            ));
        }
        
        // Validate consciousness threshold
        if self.consciousness_threshold < 0.0 || self.consciousness_threshold > 1.0 {
            return Err(ImhotepError::ConfigurationError(
                "Consciousness threshold must be between 0.0 and 1.0".to_string()
            ));
        }
        
        // Validate specialized systems
        let valid_systems = vec![
            "autobahn", "heihachi", "helicopter", "izinyoka",
            "kwasa_kwasa", "four_sided_triangle", "bene_gesserit", "nebuchadnezzar"
        ];
        
        for system in &self.specialized_systems {
            if !valid_systems.contains(&system.as_str()) {
                return Err(ImhotepError::ConfigurationError(
                    format!("Unknown specialized system: {}", system)
                ));
            }
        }
        
        // Validate quantum parameters
        self.quantum_params.validate()?;
        
        Ok(())
    }
    
    /// Get consciousness configuration
    pub fn get_consciousness_config(&self) -> ConsciousnessConfig {
        ConsciousnessConfig {
            quantum_enhancement: self.quantum_enhancement.clone(),
            fire_wavelength: self.fire_wavelength,
            consciousness_threshold: self.consciousness_threshold,
            specialized_systems: self.specialized_systems.clone(),
            authenticity_validation: self.authenticity_validation,
        }
    }
}

impl QuantumParameters {
    /// Validate quantum parameters
    pub fn validate(&self) -> ImhotepResult<()> {
        // Validate proton tunneling
        if self.proton_tunneling.probability_threshold < 0.0 || self.proton_tunneling.probability_threshold > 1.0 {
            return Err(ImhotepError::ConfigurationError(
                "Proton tunneling probability threshold must be between 0.0 and 1.0".to_string()
            ));
        }
        
        // Validate coherence parameters
        if self.coherence_maintenance.coherence_time <= 0.0 {
            return Err(ImhotepError::ConfigurationError(
                "Coherence time must be positive".to_string()
            ));
        }
        
        // Validate fire wavelength optimization
        if self.fire_wavelength_optimization.target_wavelength < 400.0 || 
           self.fire_wavelength_optimization.target_wavelength > 800.0 {
            return Err(ImhotepError::ConfigurationError(
                "Target wavelength must be in visible light range (400-800 nm)".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Consciousness configuration (subset of ImhotepConfig)
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    
    /// Fire wavelength for consciousness substrate activation (nm)
    pub fire_wavelength: f64,
    
    /// Consciousness authenticity threshold (0.0 - 1.0)
    pub consciousness_threshold: f64,
    
    /// Enabled specialized systems
    pub specialized_systems: Vec<String>,
    
    /// Enable consciousness authenticity validation
    pub authenticity_validation: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = ImhotepConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.fire_wavelength, 650.3);
        assert_eq!(config.consciousness_threshold, 0.85);
        assert_eq!(config.specialized_systems.len(), 8);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = ImhotepConfig::default();
        
        // Test invalid fire wavelength
        config.fire_wavelength = 1000.0;
        assert!(config.validate().is_err());
        
        // Test invalid consciousness threshold
        config.fire_wavelength = 650.3;
        config.consciousness_threshold = 1.5;
        assert!(config.validate().is_err());
        
        // Test invalid specialized system
        config.consciousness_threshold = 0.85;
        config.specialized_systems.push("invalid_system".to_string());
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_quantum_enhancement_levels() {
        let minimal = QuantumEnhancementLevel::Minimal;
        let maximum = QuantumEnhancementLevel::Maximum;
        
        assert_ne!(minimal, maximum);
        
        let custom = QuantumEnhancementLevel::Custom(CustomQuantumConfig {
            ion_field_intensity: 0.8,
            coherence_level: 0.9,
            enaqt_depth: 5,
            oscillation_coupling: 0.7,
        });
        
        match custom {
            QuantumEnhancementLevel::Custom(config) => {
                assert_eq!(config.ion_field_intensity, 0.8);
            },
            _ => panic!("Expected custom quantum config"),
        }
    }
} 