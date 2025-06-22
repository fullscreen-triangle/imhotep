//! Error handling for the Imhotep Framework
//! 
//! This module provides comprehensive error types for all components of the
//! consciousness simulation framework, enabling precise error reporting and
//! debugging across quantum processing, specialized systems, and Turbulence
//! language execution.

use std::fmt;
use thiserror::Error;

/// Main error type for the Imhotep Framework
#[derive(Error, Debug)]
pub enum ImhotepError {
    /// Consciousness simulation initialization error
    #[error("Consciousness initialization failed: {0}")]
    ConsciousnessInitializationError(String),
    
    /// Consciousness simulation runtime error
    #[error("Consciousness simulation failed: {0}")]
    ConsciousnessSimulationError(String),
    
    /// Consciousness authenticity validation error
    #[error("Authenticity validation failed: {0}")]
    AuthenticityValidationError(String),
    
    /// Quantum processing error
    #[error("Quantum processing failed: {0}")]
    QuantumProcessingError(String),
    
    /// Quantum membrane initialization error
    #[error("Quantum membrane initialization failed: {0}")]
    QuantumMembraneError(String),
    
    /// Ion field processing error
    #[error("Ion field processing failed: {0}")]
    IonFieldProcessingError(String),
    
    /// Fire wavelength coupling error
    #[error("Fire wavelength coupling failed: {0}")]
    FireWavelengthError(String),
    
    /// ENAQT processing error
    #[error("ENAQT processing failed: {0}")]
    ENAQTProcessingError(String),
    
    /// Specialized system error
    #[error("Specialized system '{system}' failed: {message}")]
    SpecializedSystemError {
        /// System name
        system: String,
        /// Error message
        message: String,
    },
    
    /// Autobahn RAG system error
    #[error("Autobahn RAG system failed: {0}")]
    AutobahnError(String),
    
    /// Heihachi fire emotion system error
    #[error("Heihachi fire emotion system failed: {0}")]
    HeihachiError(String),
    
    /// Helicopter visual understanding system error
    #[error("Helicopter visual understanding system failed: {0}")]
    HelicopterError(String),
    
    /// Izinyoka metacognitive system error
    #[error("Izinyoka metacognitive system failed: {0}")]
    IzinyokaError(String),
    
    /// KwasaKwasa semantic system error
    #[error("KwasaKwasa semantic system failed: {0}")]
    KwasaKwasaError(String),
    
    /// Four-sided triangle optimization system error
    #[error("Four-sided triangle optimization system failed: {0}")]
    FourSidedTriangleError(String),
    
    /// Bene Gesserit membrane system error
    #[error("Bene Gesserit membrane system failed: {0}")]
    BeneGesseritError(String),
    
    /// Nebuchadnezzar circuits system error
    #[error("Nebuchadnezzar circuits system failed: {0}")]
    NebuchadnezzarError(String),
    
    /// Cross-modal integration error
    #[error("Cross-modal integration failed: {0}")]
    CrossModalIntegrationError(String),
    
    /// Global workspace architecture error
    #[error("Global workspace architecture failed: {0}")]
    GlobalWorkspaceError(String),
    
    /// Unified consciousness state error
    #[error("Unified consciousness state failed: {0}")]
    UnifiedConsciousnessError(String),
    
    /// Turbulence language compilation error
    #[error("Turbulence compilation failed: {0}")]
    TurbulenceCompilationError(String),
    
    /// Turbulence language execution error
    #[error("Turbulence execution failed: {0}")]
    TurbulenceExecutionError(String),
    
    /// Turbulence parsing error
    #[error("Turbulence parsing failed at line {line}, column {column}: {message}")]
    TurbulenceParsingError {
        /// Line number
        line: usize,
        /// Column number
        column: usize,
        /// Error message
        message: String,
    },
    
    /// Turbulence semantic analysis error
    #[error("Turbulence semantic analysis failed: {0}")]
    TurbulenceSemanticError(String),
    
    /// External system orchestration error
    #[error("External system orchestration failed: {0}")]
    ExternalSystemError(String),
    
    /// Lavoisier R integration error
    #[error("Lavoisier R integration failed: {0}")]
    LavoisierError(String),
    
    /// Database consciousness APIs error
    #[error("Database consciousness APIs failed: {0}")]
    DatabaseConsciousnessError(String),
    
    /// Literature consciousness corpus error
    #[error("Literature consciousness corpus failed: {0}")]
    LiteratureConsciousnessError(String),
    
    /// Clinical validation systems error
    #[error("Clinical validation systems failed: {0}")]
    ClinicalValidationError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Validation error
    #[error("Validation failed: {0}")]
    ValidationError(String),
    
    /// Metrics collection error
    #[error("Metrics collection failed: {0}")]
    MetricsError(String),
    
    /// File system error
    #[error("File system error: {0}")]
    FileSystemError(String),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    /// Resource exhaustion error
    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustionError {
        /// Resource type
        resource: String,
    },
    
    /// Timeout error
    #[error("Operation timed out after {duration_ms}ms: {operation}")]
    TimeoutError {
        /// Operation that timed out
        operation: String,
        /// Duration in milliseconds
        duration_ms: u64,
    },
    
    /// Concurrent access error
    #[error("Concurrent access error: {0}")]
    ConcurrencyError(String),
    
    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationError(String),
    
    /// Hardware compatibility error
    #[error("Hardware compatibility error: {0}")]
    HardwareCompatibilityError(String),
    
    /// Security error
    #[error("Security error: {0}")]
    SecurityError(String),
    
    /// Permission error
    #[error("Permission denied: {0}")]
    PermissionError(String),
    
    /// Internal error (should not happen in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),
    
    /// Standard IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// TOML parsing error
    #[error("TOML error: {0}")]
    TomlError(#[from] toml::de::Error),
    
    /// Regex error
    #[error("Regex error: {0}")]
    RegexError(String),
    
    /// URL parsing error
    #[error("URL parsing error: {0}")]
    UrlParsingError(String),
    
    /// HTTP client error
    #[error("HTTP client error: {0}")]
    HttpClientError(String),
    
    /// Async runtime error
    #[error("Async runtime error: {0}")]
    AsyncRuntimeError(String),
}

/// Result type for Imhotep Framework operations
pub type ImhotepResult<T> = Result<T, ImhotepError>;

/// Error context for enhanced debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Component where error occurred
    pub component: String,
    
    /// Operation being performed
    pub operation: String,
    
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            component: component.to_string(),
            operation: operation.to_string(),
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            stack_trace: None,
        }
    }
    
    /// Add context information
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Add stack trace
    pub fn with_stack_trace(mut self, stack_trace: String) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }
}

/// Enhanced error with context
#[derive(Debug)]
pub struct ContextualError {
    /// The underlying error
    pub error: ImhotepError,
    
    /// Error context
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}::{}] {}", self.context.component, self.context.operation, self.error)?;
        
        if !self.context.context.is_empty() {
            write!(f, " (Context: ")?;
            for (key, value) in &self.context.context {
                write!(f, "{}={}, ", key, value)?;
            }
            write!(f, ")")?;
        }
        
        write!(f, " at {}", self.context.timestamp.format("%Y-%m-%d %H:%M:%S UTC"))
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue
    Low,
    /// Medium severity - operation should be retried
    Medium,
    /// High severity - operation should be aborted
    High,
    /// Critical severity - system should be shut down
    Critical,
}

impl ImhotepError {
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical errors
            ImhotepError::MemoryAllocationError(_) => ErrorSeverity::Critical,
            ImhotepError::HardwareCompatibilityError(_) => ErrorSeverity::Critical,
            ImhotepError::SecurityError(_) => ErrorSeverity::Critical,
            ImhotepError::InternalError(_) => ErrorSeverity::Critical,
            
            // High severity errors
            ImhotepError::ConsciousnessInitializationError(_) => ErrorSeverity::High,
            ImhotepError::QuantumMembraneError(_) => ErrorSeverity::High,
            ImhotepError::ResourceExhaustionError { .. } => ErrorSeverity::High,
            ImhotepError::ConcurrencyError(_) => ErrorSeverity::High,
            
            // Medium severity errors
            ImhotepError::ConsciousnessSimulationError(_) => ErrorSeverity::Medium,
            ImhotepError::QuantumProcessingError(_) => ErrorSeverity::Medium,
            ImhotepError::SpecializedSystemError { .. } => ErrorSeverity::Medium,
            ImhotepError::CrossModalIntegrationError(_) => ErrorSeverity::Medium,
            ImhotepError::TurbulenceCompilationError(_) => ErrorSeverity::Medium,
            ImhotepError::TurbulenceExecutionError(_) => ErrorSeverity::Medium,
            ImhotepError::ExternalSystemError(_) => ErrorSeverity::Medium,
            ImhotepError::TimeoutError { .. } => ErrorSeverity::Medium,
            
            // Low severity errors
            ImhotepError::AuthenticityValidationError(_) => ErrorSeverity::Low,
            ImhotepError::ValidationError(_) => ErrorSeverity::Low,
            ImhotepError::MetricsError(_) => ErrorSeverity::Low,
            ImhotepError::ConfigurationError(_) => ErrorSeverity::Low,
            ImhotepError::NetworkError(_) => ErrorSeverity::Low,
            ImhotepError::HttpClientError(_) => ErrorSeverity::Low,
            
            // Variable severity based on context
            _ => ErrorSeverity::Medium,
        }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.severity() {
            ErrorSeverity::Low | ErrorSeverity::Medium => true,
            ErrorSeverity::High | ErrorSeverity::Critical => false,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> &'static str {
        match self {
            ImhotepError::ConsciousnessInitializationError(_) |
            ImhotepError::ConsciousnessSimulationError(_) |
            ImhotepError::AuthenticityValidationError(_) => "consciousness",
            
            ImhotepError::QuantumProcessingError(_) |
            ImhotepError::QuantumMembraneError(_) |
            ImhotepError::IonFieldProcessingError(_) |
            ImhotepError::FireWavelengthError(_) |
            ImhotepError::ENAQTProcessingError(_) => "quantum",
            
            ImhotepError::SpecializedSystemError { .. } |
            ImhotepError::AutobahnError(_) |
            ImhotepError::HeihachiError(_) |
            ImhotepError::HelicopterError(_) |
            ImhotepError::IzinyokaError(_) |
            ImhotepError::KwasaKwasaError(_) |
            ImhotepError::FourSidedTriangleError(_) |
            ImhotepError::BeneGesseritError(_) |
            ImhotepError::NebuchadnezzarError(_) => "specialized_systems",
            
            ImhotepError::CrossModalIntegrationError(_) |
            ImhotepError::GlobalWorkspaceError(_) |
            ImhotepError::UnifiedConsciousnessError(_) => "cross_modal",
            
            ImhotepError::TurbulenceCompilationError(_) |
            ImhotepError::TurbulenceExecutionError(_) |
            ImhotepError::TurbulenceParsingError { .. } |
            ImhotepError::TurbulenceSemanticError(_) => "turbulence",
            
            ImhotepError::ExternalSystemError(_) |
            ImhotepError::LavoisierError(_) |
            ImhotepError::DatabaseConsciousnessError(_) |
            ImhotepError::LiteratureConsciousnessError(_) |
            ImhotepError::ClinicalValidationError(_) => "external_systems",
            
            ImhotepError::ConfigurationError(_) => "configuration",
            ImhotepError::ValidationError(_) => "validation",
            ImhotepError::MetricsError(_) => "metrics",
            
            ImhotepError::FileSystemError(_) |
            ImhotepError::IoError(_) => "filesystem",
            
            ImhotepError::NetworkError(_) |
            ImhotepError::HttpClientError(_) => "network",
            
            ImhotepError::DatabaseError(_) => "database",
            
            ImhotepError::SerializationError(_) |
            ImhotepError::DeserializationError(_) |
            ImhotepError::JsonError(_) |
            ImhotepError::TomlError(_) => "serialization",
            
            ImhotepError::ResourceExhaustionError { .. } |
            ImhotepError::MemoryAllocationError(_) => "resources",
            
            ImhotepError::TimeoutError { .. } => "timeout",
            ImhotepError::ConcurrencyError(_) => "concurrency",
            ImhotepError::HardwareCompatibilityError(_) => "hardware",
            ImhotepError::SecurityError(_) => "security",
            ImhotepError::PermissionError(_) => "permissions",
            ImhotepError::AsyncRuntimeError(_) => "async",
            
            _ => "general",
        }
    }
    
    /// Create error with context
    pub fn with_context(self, context: ErrorContext) -> ContextualError {
        ContextualError {
            error: self,
            context,
        }
    }
}

/// Macro for creating contextual errors
#[macro_export]
macro_rules! contextual_error {
    ($error:expr, $component:expr, $operation:expr) => {
        $error.with_context(ErrorContext::new($component, $operation))
    };
    ($error:expr, $component:expr, $operation:expr, $($key:expr => $value:expr),+) => {
        {
            let mut context = ErrorContext::new($component, $operation);
            $(
                context = context.with_context($key, $value);
            )+
            $error.with_context(context)
        }
    };
}

/// Macro for creating ImhotepError with formatted message
#[macro_export]
macro_rules! imhotep_error {
    ($variant:ident, $($arg:tt)*) => {
        ImhotepError::$variant(format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        let error = ImhotepError::ConsciousnessInitializationError("test".to_string());
        assert_eq!(error.severity(), ErrorSeverity::High);
        assert!(!error.is_recoverable());
    }
    
    #[test]
    fn test_error_category() {
        let error = ImhotepError::QuantumProcessingError("test".to_string());
        assert_eq!(error.category(), "quantum");
    }
    
    #[test]
    fn test_contextual_error() {
        let error = ImhotepError::ConsciousnessSimulationError("test".to_string());
        let context = ErrorContext::new("consciousness", "simulation")
            .with_context("input_size", "1024")
            .with_context("threshold", "0.85");
        
        let contextual_error = error.with_context(context);
        let error_string = contextual_error.to_string();
        
        assert!(error_string.contains("consciousness::simulation"));
        assert!(error_string.contains("input_size=1024"));
        assert!(error_string.contains("threshold=0.85"));
    }
    
    #[test]
    fn test_error_macros() {
        let error = imhotep_error!(ConsciousnessSimulationError, "Failed with code: {}", 42);
        match error {
            ImhotepError::ConsciousnessSimulationError(msg) => {
                assert_eq!(msg, "Failed with code: 42");
            },
            _ => panic!("Wrong error type"),
        }
    }
} 