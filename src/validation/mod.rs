//! Validation Module
//! 
//! This module implements validation capabilities for consciousness authenticity
//! and simulation quality assurance.

use serde::{Deserialize, Serialize};
use crate::error::{ImhotepError, ImhotepResult};

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub is_valid: bool,
    pub confidence: f64,
    pub validation_score: f64,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub error_type: ValidationErrorType,
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    AuthenticityFailure,
    CoherenceFailure,
    QualityFailure,
}

impl ValidationResults {
    pub fn new(is_valid: bool, confidence: f64, validation_score: f64) -> Self {
        Self {
            is_valid,
            confidence,
            validation_score,
        }
    }
}

impl ValidationError {
    pub fn new(message: String, error_type: ValidationErrorType) -> Self {
        Self {
            message,
            error_type,
        }
    }
} 