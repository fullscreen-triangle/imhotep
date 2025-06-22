//! Ion Field Dynamics Processor
//! 
//! This module implements ion field dynamics for consciousness simulation,
//! managing ionic flows and electric field dynamics that support
//! consciousness-enhanced quantum transport and neural coherence.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use std::collections::HashMap;

use crate::error::{ImhotepError, ImhotepResult};
use crate::quantum::{QuantumState, QuantumParameters, IonFieldResults};

/// Ion field dynamics processor
pub struct IonFieldProcessor {
    /// Field processing parameters
    parameters: IonFieldParameters,
    
    /// Electric field distribution
    electric_field: DMatrix<Complex64>,
    
    /// Ion concentration gradients
    ion_concentrations: HashMap<IonType, DVector<f64>>,
    
    /// Membrane potential distribution
    membrane_potentials: DVector<f64>,
    
    /// Ion channel states
    ion_channels: Vec<IonChannel>,
    
    /// Field dynamics history
    dynamics_history: Vec<FieldDynamicsSnapshot>,
}

/// Ion field processing parameters
#[derive(Debug, Clone)]
pub struct IonFieldParameters {
    /// System temperature (K)
    pub temperature: f64,
    
    /// Membrane permeability (cm/s)
    pub membrane_permeability: f64,
    
    /// Ionic strength (M)
    pub ionic_strength: f64,
    
    /// Electric field strength (V/m)
    pub field_strength: f64,
    
    /// Consciousness coupling factor
    pub consciousness_coupling: f64,
    
    /// Quantum enhancement level
    pub quantum_enhancement: f64,
    
    /// Diffusion coefficient (cm²/s)
    pub diffusion_coefficient: f64,
}

/// Ion types in the system
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum IonType {
    /// Sodium ions (Na+)
    Sodium,
    
    /// Potassium ions (K+)
    Potassium,
    
    /// Calcium ions (Ca2+)
    Calcium,
    
    /// Chloride ions (Cl-)
    Chloride,
    
    /// Magnesium ions (Mg2+)
    Magnesium,
    
    /// Consciousness-coupled exotic ions
    ConsciousnessIons,
}

/// Ion channel representation
#[derive(Debug, Clone)]
pub struct IonChannel {
    /// Channel identifier
    pub id: String,
    
    /// Channel type
    pub channel_type: IonChannelType,
    
    /// Spatial position
    pub position: usize,
    
    /// Conductance (S)
    pub conductance: f64,
    
    /// Gating state (0.0 - 1.0)
    pub gating_state: f64,
    
    /// Selectivity for different ions
    pub selectivity: HashMap<IonType, f64>,
    
    /// Consciousness modulation factor
    pub consciousness_modulation: f64,
}

/// Ion channel types
#[derive(Debug, Clone)]
pub enum IonChannelType {
    /// Voltage-gated sodium channel
    VoltageGatedSodium,
    
    /// Voltage-gated potassium channel
    VoltageGatedPotassium,
    
    /// Voltage-gated calcium channel
    VoltageGatedCalcium,
    
    /// Ligand-gated channel
    LigandGated {
        ligand_type: String,
    },
    
    /// Mechanosensitive channel
    Mechanosensitive,
    
    /// Consciousness-gated channel
    ConsciousnessGated {
        consciousness_threshold: f64,
    },
}

/// Field dynamics snapshot
#[derive(Debug, Clone)]
pub struct FieldDynamicsSnapshot {
    /// Timestamp (ns)
    pub timestamp: f64,
    
    /// Average electric field strength
    pub avg_field_strength: f64,
    
    /// Membrane potential variance
    pub membrane_potential_variance: f64,
    
    /// Ion flux rates
    pub ion_flux_rates: HashMap<IonType, f64>,
    
    /// Consciousness coupling strength
    pub consciousness_coupling: f64,
    
    /// Quantum coherence level
    pub quantum_coherence: f64,
}

/// Electrochemical gradient
#[derive(Debug, Clone)]
pub struct ElectrochemicalGradient {
    /// Chemical gradient (concentration difference)
    pub chemical_gradient: f64,
    
    /// Electrical gradient (potential difference)
    pub electrical_gradient: f64,
    
    /// Total electrochemical potential
    pub electrochemical_potential: f64,
    
    /// Driving force for ion movement
    pub driving_force: f64,
}

impl IonFieldProcessor {
    /// Create new ion field processor
    pub fn new(parameters: &QuantumParameters) -> ImhotepResult<Self> {
        let ion_params = IonFieldParameters::from_quantum_parameters(parameters);
        let system_size = 16; // Default system size
        
        let electric_field = Self::initialize_electric_field(system_size, &ion_params)?;
        let ion_concentrations = Self::initialize_ion_concentrations(system_size)?;
        let membrane_potentials = Self::initialize_membrane_potentials(system_size, &ion_params)?;
        let ion_channels = Self::initialize_ion_channels(system_size, &ion_params)?;
        let dynamics_history = Vec::new();
        
        Ok(Self {
            parameters: ion_params,
            electric_field,
            ion_concentrations,
            membrane_potentials,
            ion_channels,
            dynamics_history,
        })
    }
    
    /// Process ion field dynamics
    pub async fn process_ion_field_dynamics(
        &mut self,
        state: &QuantumState,
    ) -> ImhotepResult<IonFieldResults> {
        // Update electric field based on quantum state
        self.update_electric_field(state).await?;
        
        // Calculate ion transport
        let transport_efficiency = self.calculate_ion_transport_efficiency()?;
        
        // Update membrane potentials
        self.update_membrane_potentials().await?;
        
        // Calculate field stability
        let field_stability = self.calculate_field_stability()?;
        
        // Assess consciousness coupling
        let consciousness_coupling = self.calculate_consciousness_coupling(state)?;
        
        // Calculate quantum enhancement
        let quantum_enhancement = self.calculate_quantum_enhancement(state)?;
        
        // Record dynamics snapshot
        let snapshot = self.create_dynamics_snapshot(
            transport_efficiency,
            consciousness_coupling,
            quantum_enhancement,
        )?;
        self.dynamics_history.push(snapshot);
        
        Ok(IonFieldResults {
            transport_efficiency,
            field_stability,
            consciousness_coupling,
            quantum_enhancement,
        })
    }
    
    /// Initialize electric field distribution
    fn initialize_electric_field(system_size: usize, params: &IonFieldParameters) -> ImhotepResult<DMatrix<Complex64>> {
        let mut field = DMatrix::zeros(system_size, system_size);
        
        // Create realistic electric field distribution
        for i in 0..system_size {
            for j in 0..system_size {
                if i == j {
                    // Self-field (membrane potential)
                    let membrane_field = params.field_strength * 
                        (1.0 + params.consciousness_coupling * 
                         (i as f64 / system_size as f64 * std::f64::consts::PI).sin());
                    field[(i, j)] = Complex64::new(membrane_field, 0.0);
                } else {
                    // Inter-site field coupling
                    let distance = ((i as f64 - j as f64).powi(2)).sqrt();
                    let coupling = params.field_strength / (1.0 + distance) * 
                                  params.quantum_enhancement;
                    
                    // Phase from ion movement
                    let phase = 2.0 * std::f64::consts::PI * distance / 10.0; // 10 unit characteristic length
                    
                    field[(i, j)] = Complex64::new(
                        coupling * phase.cos(),
                        coupling * phase.sin()
                    );
                }
            }
        }
        
        Ok(field)
    }
    
    /// Initialize ion concentrations
    fn initialize_ion_concentrations(system_size: usize) -> ImhotepResult<HashMap<IonType, DVector<f64>>> {
        let mut concentrations = HashMap::new();
        
        // Physiological ion concentrations (mM)
        let ion_configs = vec![
            (IonType::Sodium, 145.0),      // Extracellular Na+
            (IonType::Potassium, 5.0),     // Extracellular K+
            (IonType::Calcium, 2.5),       // Extracellular Ca2+
            (IonType::Chloride, 110.0),    // Extracellular Cl-
            (IonType::Magnesium, 1.0),     // Extracellular Mg2+
            (IonType::ConsciousnessIons, 0.1), // Exotic consciousness-coupled ions
        ];
        
        for (ion_type, base_concentration) in ion_configs {
            let mut concentration_vector = DVector::zeros(system_size);
            
            for i in 0..system_size {
                // Add spatial variation and consciousness coupling
                let spatial_factor = 1.0 + 0.2 * (i as f64 / system_size as f64 * 
                                                  2.0 * std::f64::consts::PI).sin();
                
                let consciousness_factor = if matches!(ion_type, IonType::ConsciousnessIons) {
                    1.0 + (i as f64 / system_size as f64 * std::f64::consts::PI).sin().abs()
                } else {
                    1.0
                };
                
                concentration_vector[i] = base_concentration * spatial_factor * consciousness_factor;
            }
            
            concentrations.insert(ion_type, concentration_vector);
        }
        
        Ok(concentrations)
    }
    
    /// Initialize membrane potentials
    fn initialize_membrane_potentials(system_size: usize, params: &IonFieldParameters) -> ImhotepResult<DVector<f64>> {
        let mut potentials = DVector::zeros(system_size);
        
        // Resting membrane potential around -70 mV with variations
        let base_potential = -70.0; // mV
        
        for i in 0..system_size {
            let spatial_variation = 10.0 * (i as f64 / system_size as f64 * 
                                           std::f64::consts::PI).sin();
            
            let consciousness_modulation = params.consciousness_coupling * 5.0 * 
                                          (i as f64 / system_size as f64 * 
                                           2.0 * std::f64::consts::PI).cos();
            
            potentials[i] = base_potential + spatial_variation + consciousness_modulation;
        }
        
        Ok(potentials)
    }
    
    /// Initialize ion channels
    fn initialize_ion_channels(system_size: usize, params: &IonFieldParameters) -> ImhotepResult<Vec<IonChannel>> {
        let mut channels = Vec::new();
        
        // Create different types of ion channels
        for i in 0..system_size {
            // Voltage-gated sodium channels
            let na_channel = IonChannel {
                id: format!("Na_channel_{}", i),
                channel_type: IonChannelType::VoltageGatedSodium,
                position: i,
                conductance: 20e-12, // 20 pS
                gating_state: 0.1,   // Mostly closed at rest
                selectivity: {
                    let mut sel = HashMap::new();
                    sel.insert(IonType::Sodium, 1.0);
                    sel.insert(IonType::Potassium, 0.1);
                    sel
                },
                consciousness_modulation: params.consciousness_coupling,
            };
            channels.push(na_channel);
            
            // Voltage-gated potassium channels
            let k_channel = IonChannel {
                id: format!("K_channel_{}", i),
                channel_type: IonChannelType::VoltageGatedPotassium,
                position: i,
                conductance: 10e-12, // 10 pS
                gating_state: 0.3,   // Partially open at rest
                selectivity: {
                    let mut sel = HashMap::new();
                    sel.insert(IonType::Potassium, 1.0);
                    sel.insert(IonType::Sodium, 0.05);
                    sel
                },
                consciousness_modulation: params.consciousness_coupling * 0.8,
            };
            channels.push(k_channel);
            
            // Consciousness-gated channels (every 4th position)
            if i % 4 == 0 {
                let consciousness_channel = IonChannel {
                    id: format!("Consciousness_channel_{}", i),
                    channel_type: IonChannelType::ConsciousnessGated {
                        consciousness_threshold: 0.5,
                    },
                    position: i,
                    conductance: 5e-12, // 5 pS
                    gating_state: params.consciousness_coupling.min(1.0),
                    selectivity: {
                        let mut sel = HashMap::new();
                        sel.insert(IonType::ConsciousnessIons, 1.0);
                        sel.insert(IonType::Calcium, 0.3);
                        sel
                    },
                    consciousness_modulation: params.consciousness_coupling * 2.0,
                };
                channels.push(consciousness_channel);
            }
        }
        
        Ok(channels)
    }
    
    /// Update electric field based on quantum state
    async fn update_electric_field(&mut self, state: &QuantumState) -> ImhotepResult<()> {
        let system_size = self.electric_field.nrows();
        
        for i in 0..system_size {
            for j in 0..system_size {
                if i < state.dimension && j < state.dimension {
                    // Quantum state influence on electric field
                    let quantum_amplitude = state.state_vector[i] * state.state_vector[j].conj();
                    let field_enhancement = 1.0 + self.parameters.quantum_enhancement * 
                                          quantum_amplitude.norm();
                    
                    self.electric_field[(i, j)] *= field_enhancement;
                    
                    // Consciousness coupling enhancement
                    let consciousness_phase = Complex64::new(
                        0.0, 
                        self.parameters.consciousness_coupling * quantum_amplitude.arg()
                    ).exp();
                    
                    self.electric_field[(i, j)] *= consciousness_phase;
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate ion transport efficiency
    fn calculate_ion_transport_efficiency(&self) -> ImhotepResult<f64> {
        let mut total_efficiency = 0.0;
        let mut channel_count = 0;
        
        // Calculate efficiency based on ion channel states
        for channel in &self.ion_channels {
            let channel_efficiency = channel.gating_state * channel.conductance * 
                                   (1.0 + channel.consciousness_modulation);
            total_efficiency += channel_efficiency;
            channel_count += 1;
        }
        
        if channel_count > 0 {
            total_efficiency /= channel_count as f64;
        }
        
        // Normalize to 0-1 range
        total_efficiency = (total_efficiency / 20e-12).min(1.0); // Normalize by typical conductance
        
        // Electric field contribution
        let field_magnitude = self.electric_field.norm() / self.electric_field.len() as f64;
        let field_efficiency = (field_magnitude / self.parameters.field_strength).min(1.0);
        
        // Combined efficiency
        let combined_efficiency = (total_efficiency + field_efficiency) / 2.0;
        
        Ok(combined_efficiency)
    }
    
    /// Update membrane potentials
    async fn update_membrane_potentials(&mut self) -> ImhotepResult<()> {
        let system_size = self.membrane_potentials.len();
        
        for i in 0..system_size {
            // Calculate new membrane potential based on ion concentrations and field
            let mut new_potential = 0.0;
            
            // Nernst potential contributions
            if let (Some(na_conc), Some(k_conc)) = (
                self.ion_concentrations.get(&IonType::Sodium),
                self.ion_concentrations.get(&IonType::Potassium)
            ) {
                // Simplified Goldman-Hodgkin-Katz equation
                let rt_f = 26.7; // RT/F at 37°C in mV
                
                let na_contrib = rt_f * (na_conc[i] / 15.0).ln(); // Intracellular Na+ ~15 mM
                let k_contrib = rt_f * (140.0 / k_conc[i]).ln();  // Intracellular K+ ~140 mM
                
                new_potential = -70.0 + 0.1 * na_contrib + 0.9 * k_contrib; // Weighted by permeability
            }
            
            // Electric field influence
            let field_influence = self.electric_field[(i, i)].re * 0.1; // Scale factor
            new_potential += field_influence;
            
            // Consciousness coupling
            let consciousness_influence = self.parameters.consciousness_coupling * 10.0 * 
                                        (i as f64 / system_size as f64 * std::f64::consts::PI).sin();
            new_potential += consciousness_influence;
            
            self.membrane_potentials[i] = new_potential;
        }
        
        Ok(())
    }
    
    /// Calculate field stability
    fn calculate_field_stability(&self) -> ImhotepResult<f64> {
        if self.dynamics_history.len() < 2 {
            return Ok(1.0); // Assume stable if no history
        }
        
        // Calculate variance in field strength over recent history
        let recent_history = &self.dynamics_history[self.dynamics_history.len().saturating_sub(10)..];
        
        let mean_field = recent_history.iter()
            .map(|snapshot| snapshot.avg_field_strength)
            .sum::<f64>() / recent_history.len() as f64;
        
        let variance = recent_history.iter()
            .map(|snapshot| (snapshot.avg_field_strength - mean_field).powi(2))
            .sum::<f64>() / recent_history.len() as f64;
        
        let stability = 1.0 / (1.0 + variance); // Higher variance = lower stability
        
        Ok(stability.clamp(0.0, 1.0))
    }
    
    /// Calculate consciousness coupling
    fn calculate_consciousness_coupling(&self, state: &QuantumState) -> ImhotepResult<f64> {
        let mut coupling = 0.0;
        
        // Quantum coherence contribution
        let coherence = self.calculate_quantum_coherence(state)?;
        coupling += coherence * self.parameters.consciousness_coupling;
        
        // Ion channel consciousness modulation
        let mut consciousness_channel_activity = 0.0;
        let mut consciousness_channel_count = 0;
        
        for channel in &self.ion_channels {
            if matches!(channel.channel_type, IonChannelType::ConsciousnessGated { .. }) {
                consciousness_channel_activity += channel.gating_state * channel.consciousness_modulation;
                consciousness_channel_count += 1;
            }
        }
        
        if consciousness_channel_count > 0 {
            consciousness_channel_activity /= consciousness_channel_count as f64;
            coupling += consciousness_channel_activity * 0.3;
        }
        
        // Consciousness ion concentration
        if let Some(consciousness_ions) = self.ion_concentrations.get(&IonType::ConsciousnessIons) {
            let avg_consciousness_concentration = consciousness_ions.sum() / consciousness_ions.len() as f64;
            coupling += (avg_consciousness_concentration / 0.1) * 0.2; // Normalize by base concentration
        }
        
        Ok(coupling.min(1.0))
    }
    
    /// Calculate quantum enhancement
    fn calculate_quantum_enhancement(&self, state: &QuantumState) -> ImhotepResult<f64> {
        let mut enhancement = 0.0;
        
        // Quantum coherence enhancement
        let coherence = self.calculate_quantum_coherence(state)?;
        enhancement += coherence * self.parameters.quantum_enhancement;
        
        // Electric field quantum coupling
        let field_quantum_coupling = self.calculate_field_quantum_coupling(state)?;
        enhancement += field_quantum_coupling * 0.3;
        
        // Ion transport quantum effects
        let transport_quantum_effects = self.calculate_transport_quantum_effects(state)?;
        enhancement += transport_quantum_effects * 0.2;
        
        Ok(enhancement.min(2.0)) // Cap at 2x enhancement
    }
    
    /// Calculate quantum coherence
    fn calculate_quantum_coherence(&self, state: &QuantumState) -> ImhotepResult<f64> {
        // Calculate purity as coherence measure
        let density_squared = &state.density_matrix * &state.density_matrix;
        let purity = density_squared.trace().re;
        
        Ok(purity.clamp(0.0, 1.0))
    }
    
    /// Calculate field-quantum coupling
    fn calculate_field_quantum_coupling(&self, state: &QuantumState) -> ImhotepResult<f64> {
        let mut coupling = 0.0;
        let system_size = self.electric_field.nrows().min(state.dimension);
        
        for i in 0..system_size {
            let field_strength = self.electric_field[(i, i)].norm();
            let quantum_amplitude = state.state_vector[i].norm();
            coupling += field_strength * quantum_amplitude;
        }
        
        coupling /= system_size as f64;
        
        // Normalize by typical field strength
        coupling /= self.parameters.field_strength;
        
        Ok(coupling.min(1.0))
    }
    
    /// Calculate transport quantum effects
    fn calculate_transport_quantum_effects(&self, state: &QuantumState) -> ImhotepResult<f64> {
        let mut quantum_effects = 0.0;
        
        // Quantum tunneling probability for ion transport
        for channel in &self.ion_channels {
            if channel.position < state.dimension {
                let quantum_amplitude = state.state_vector[channel.position].norm();
                let tunneling_probability = quantum_amplitude * self.parameters.quantum_enhancement;
                quantum_effects += tunneling_probability * channel.gating_state;
            }
        }
        
        quantum_effects /= self.ion_channels.len() as f64;
        
        Ok(quantum_effects.min(1.0))
    }
    
    /// Create dynamics snapshot
    fn create_dynamics_snapshot(
        &self,
        transport_efficiency: f64,
        consciousness_coupling: f64,
        quantum_enhancement: f64,
    ) -> ImhotepResult<FieldDynamicsSnapshot> {
        // Calculate average field strength
        let avg_field_strength = self.electric_field.norm() / self.electric_field.len() as f64;
        
        // Calculate membrane potential variance
        let mean_potential = self.membrane_potentials.sum() / self.membrane_potentials.len() as f64;
        let potential_variance = self.membrane_potentials.iter()
            .map(|&potential| (potential - mean_potential).powi(2))
            .sum::<f64>() / self.membrane_potentials.len() as f64;
        
        // Calculate ion flux rates
        let mut ion_flux_rates = HashMap::new();
        for (ion_type, concentrations) in &self.ion_concentrations {
            let flux_rate = concentrations.iter()
                .zip(concentrations.iter().skip(1))
                .map(|(c1, c2)| (c2 - c1).abs())
                .sum::<f64>() / (concentrations.len() - 1) as f64;
            ion_flux_rates.insert(ion_type.clone(), flux_rate);
        }
        
        Ok(FieldDynamicsSnapshot {
            timestamp: chrono::Utc::now().timestamp_nanos() as f64 / 1e9,
            avg_field_strength,
            membrane_potential_variance: potential_variance,
            ion_flux_rates,
            consciousness_coupling,
            quantum_coherence: quantum_enhancement,
        })
    }
    
    /// Update processor parameters
    pub fn update_parameters(&mut self, parameters: &QuantumParameters) -> ImhotepResult<()> {
        self.parameters = IonFieldParameters::from_quantum_parameters(parameters);
        
        // Reinitialize components with new parameters
        let system_size = self.electric_field.nrows();
        self.electric_field = Self::initialize_electric_field(system_size, &self.parameters)?;
        self.membrane_potentials = Self::initialize_membrane_potentials(system_size, &self.parameters)?;
        
        // Update ion channels
        for channel in &mut self.ion_channels {
            channel.consciousness_modulation = self.parameters.consciousness_coupling;
        }
        
        Ok(())
    }
    
    /// Check processor health
    pub fn is_healthy(&self) -> bool {
        !self.ion_channels.is_empty() &&
        !self.ion_concentrations.is_empty() &&
        self.parameters.field_strength > 0.0 &&
        self.parameters.consciousness_coupling >= 0.0
    }
    
    /// Get ion channel statistics
    pub fn get_channel_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        let mut total_conductance = 0.0;
        let mut avg_gating_state = 0.0;
        let mut consciousness_channels = 0;
        
        for channel in &self.ion_channels {
            total_conductance += channel.conductance;
            avg_gating_state += channel.gating_state;
            
            if matches!(channel.channel_type, IonChannelType::ConsciousnessGated { .. }) {
                consciousness_channels += 1;
            }
        }
        
        if !self.ion_channels.is_empty() {
            avg_gating_state /= self.ion_channels.len() as f64;
        }
        
        stats.insert("total_conductance".to_string(), total_conductance);
        stats.insert("average_gating_state".to_string(), avg_gating_state);
        stats.insert("consciousness_channels".to_string(), consciousness_channels as f64);
        stats.insert("total_channels".to_string(), self.ion_channels.len() as f64);
        
        // Membrane potential statistics
        let mean_potential = self.membrane_potentials.sum() / self.membrane_potentials.len() as f64;
        let potential_variance = self.membrane_potentials.iter()
            .map(|&potential| (potential - mean_potential).powi(2))
            .sum::<f64>() / self.membrane_potentials.len() as f64;
        
        stats.insert("mean_membrane_potential".to_string(), mean_potential);
        stats.insert("membrane_potential_variance".to_string(), potential_variance);
        
        stats
    }
}

impl IonFieldParameters {
    /// Create from quantum parameters
    pub fn from_quantum_parameters(params: &QuantumParameters) -> Self {
        Self {
            temperature: 310.0, // Body temperature (37°C)
            membrane_permeability: 1e-6, // 1 µm/s
            ionic_strength: 0.15, // 150 mM physiological
            field_strength: params.environmental_coupling * 1e6, // Convert to V/m
            consciousness_coupling: params.consciousness_enhancement,
            quantum_enhancement: params.coherence_level,
            diffusion_coefficient: 1e-9, // 1 nm²/s
        }
    }
}

impl IonType {
    /// Get ion charge
    pub fn charge(&self) -> i32 {
        match self {
            IonType::Sodium => 1,
            IonType::Potassium => 1,
            IonType::Calcium => 2,
            IonType::Chloride => -1,
            IonType::Magnesium => 2,
            IonType::ConsciousnessIons => 1, // Assume +1 for simplicity
        }
    }
    
    /// Get typical intracellular concentration (mM)
    pub fn intracellular_concentration(&self) -> f64 {
        match self {
            IonType::Sodium => 15.0,
            IonType::Potassium => 140.0,
            IonType::Calcium => 0.0001, // 100 nM
            IonType::Chloride => 10.0,
            IonType::Magnesium => 0.5,
            IonType::ConsciousnessIons => 0.01, // Very low concentration
        }
    }
    
    /// Get typical extracellular concentration (mM)
    pub fn extracellular_concentration(&self) -> f64 {
        match self {
            IonType::Sodium => 145.0,
            IonType::Potassium => 5.0,
            IonType::Calcium => 2.5,
            IonType::Chloride => 110.0,
            IonType::Magnesium => 1.0,
            IonType::ConsciousnessIons => 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ion_field_processor_creation() {
        let params = QuantumParameters::default();
        let processor = IonFieldProcessor::new(&params);
        assert!(processor.is_ok());
        
        let processor = processor.unwrap();
        assert!(processor.is_healthy());
        assert!(!processor.ion_channels.is_empty());
    }
    
    #[tokio::test]
    async fn test_ion_field_processing() {
        let params = QuantumParameters::default();
        let mut processor = IonFieldProcessor::new(&params).unwrap();
        let state = QuantumState::consciousness_optimized(16);
        
        let result = processor.process_ion_field_dynamics(&state).await;
        assert!(result.is_ok());
        
        let results = result.unwrap();
        assert!(results.transport_efficiency >= 0.0);
        assert!(results.transport_efficiency <= 1.0);
        assert!(results.field_stability >= 0.0);
        assert!(results.field_stability <= 1.0);
    }
    
    #[test]
    fn test_ion_concentrations() {
        let concentrations = IonFieldProcessor::initialize_ion_concentrations(16).unwrap();
        
        assert!(concentrations.contains_key(&IonType::Sodium));
        assert!(concentrations.contains_key(&IonType::Potassium));
        assert!(concentrations.contains_key(&IonType::ConsciousnessIons));
        
        // Check that concentrations are reasonable
        let na_conc = &concentrations[&IonType::Sodium];
        assert!(na_conc.iter().all(|&c| c > 100.0 && c < 200.0)); // Physiological range
    }
    
    #[test]
    fn test_ion_properties() {
        assert_eq!(IonType::Sodium.charge(), 1);
        assert_eq!(IonType::Calcium.charge(), 2);
        assert_eq!(IonType::Chloride.charge(), -1);
        
        assert!(IonType::Potassium.intracellular_concentration() > 100.0);
        assert!(IonType::Sodium.extracellular_concentration() > 100.0);
    }
}
