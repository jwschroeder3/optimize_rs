use std::fmt::Debug;
use std::any::type_name;

pub mod simulated_annealing;
pub mod particle_swarm;
pub mod replica_exchange;
//pub mod pidao;
//pub mod bayes_opt;

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[derive(Clone, Debug)]
//    struct System;
//    impl Objective for System {
//        fn objective(&self, param: &Vec<f64>) -> f64 {
//            rosenbrock(param)
//        }
//    }
//
//    /// Himmelblau test function (copied directly from argmin-testfunctions
//    /// source code then modified slightly)
//    ///
//    /// Defined as
//    ///
//    /// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
//    ///
//    /// where `x_i \in [-5, 5]`.
//    ///
//    /// The global minima are at
//    ///  * `f(x_1, x_2) = f(3, 2) = 0`.
//    ///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
//    ///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
//    ///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
//    fn himmelblau(param: &Vec<f64>) -> f64 {
//        assert!(param.len() == 2);
//        let (x1, x2) = (param[0], param[1]);
//        (x1.powi(2) + x2 - 11.0).powi(2)
//            + (x1 + x2.powi(2) - 7.0).powi(2)
//    }
//
//    /// Multidimensional Rosenbrock test function (copied and slightly modified from
//    /// the argmin-testfunctions source)
//    ///
//    /// Defined as
//    ///
//    /// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
//    ///
//    /// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
//    ///
//    /// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
//    pub fn rosenbrock(param: &Vec<f64>) -> f64 {
//        param.iter()
//            .zip(param.iter().skip(1))
//            .map(|(&xi, &xi1)| (1.0 - xi).powi(2) + 100.0 * (xi1 - xi.powi(2)).powi(2))
//            .sum()
//    }
//
//    fn set_up_particles<'a, T>(
//            n_particles: usize,
//            temp: f64,
//            step: &'a f64,
//            low: &'a Vec<f64>,
//            up: &'a Vec<f64>,
//            init_jitter: f64,
//            object: T,
//            method: &Method,
//    ) -> Particles<T>
//        where
//            T: Objective + Clone + std::marker::Send,
//    {
//
//        let data_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];
//
//        Particles::new(
//            n_particles,
//            data_vec,
//            low.to_vec(),
//            up.to_vec(),
//            temp,
//            *step,
//            init_jitter,
//            object,
//            method,
//        )
//    }
//
//    fn set_up_particle<T>(
//            data_vec: &Vec<f64>,
//            step: &f64,
//            low: &Vec<f64>,
//            up: &Vec<f64>,
//            object: T,
//            method: &Method
//    ) -> Particle<T>
//        where
//            T: Objective + std::marker::Send + Clone,
//    {
//
//        let temp = 2.0;
//        let particle = ParticleBuilder::new()
//            .set_data(data_vec.to_vec()).unwrap()
//            .set_lower(low.to_vec()).unwrap()
//            .set_upper(up.to_vec()).unwrap()
//            .set_objective(object)
//            .set_temperature(temp).unwrap()
//            .set_stepsize(*step).unwrap()
//            .set_method(*method)
//            .build().unwrap();
//        particle
//    }
//
//    #[test]
//    fn test_eval() {
//        let obj = System{};
//        let start_data = vec![0.0, 0.0];
//        let step = 0.0;
//        let low = vec![-5.0, -5.0];
//        let up = vec![5.0, 5.0];
//        let particle = set_up_particle(
//            &start_data,
//            &step,
//            &low,
//            &up,
//            obj,
//            &Method::SimulatedAnnealing,
//        );
//        let score = particle.evaluate();
//        // for himmelblau
//        //assert_eq!(&score, &170.0);
//        // for rosenbrock
//        assert_eq!(&score, &1.0);
//    }
//
//}

fn type_of<T>(_: &T) -> &str {
    type_name::<T>()
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    SimulatedAnnealing,
    ParticleSwarm,
    ReplicaExchange,
    BayesianOptimization,
    PIDAO,
}

#[derive(Debug)]
enum IncompleteParticleBuild {
    NoMethod,
    NoObjective,
    NoStepsize,
    NoPosition,
}

#[derive(Debug)]
enum InvalidParticleBuild {
    NoMethod,
    InvalidDataType,
}

trait Optimizer {
    fn optimize(&mut self, niter: usize) -> (Vec<f64>, f64);
}

#[derive(Debug)]
pub struct OptimizerBuilder<T> {
    method: Option<Method>,
    particle_number: Option<usize>,
    object: Option<T>,
    position: Option<Vec<f64>>,
    lower_bound: Option<Vec<f64>>,
    upper_bound: Option<Vec<f64>>,
    temperature: Option<f64>,
    stepsize: Option<f64>,
}

#[derive(Debug)]
enum IncompleteOptimizerBuild {
    NoMethod,
    NoObjective,
    NoStepsize,
    NoPosition,
}

#[derive(Debug)]
enum InvalidOptimizerBuild {
    NoMethod,
    InvalidDataType,
}

//impl<T: Objective + std::marker::Send + Clone> OptimizerBuilder<T> {
//
//    fn new() -> Self {
//        Self{
//            method: None,
//            particle_number: None,
//            object: None,
//            position: None,
//            lower_bound: None,
//            upper_bound: None,
//            temperature: None,
//            stepsize: None,
//        }
//    }
//
//    fn set_particle_number(&mut self, number: usize) -> &mut Self {
//        self.particle_number = Some(number);
//        self
//    }
//
//    fn set_method(&mut self, method: Method) -> &mut Self {
//        self.method = Some(method);
//        self
//    }
//
//    fn set_data(&mut self, data: Vec<f64>) -> &mut Self {
//        self.position = Some(data);
//        self
//    }
//
//    fn set_lower(&mut self, lower: Vec<f64>) -> &mut Self {
//        self.lower_bound = Some(lower);
//        self
//    }
//
//    fn set_upper(&mut self, upper: Vec<f64>) -> &mut Self {
//        self.upper_bound = Some(upper);
//        self
//    }
//
//    fn set_objective(&mut self, objective: T) ->
//        &mut Self
//            where
//        T: Objective + std::marker::Send
//    {
//        self.object = Some(objective);
//        self
//    }
//
//    fn set_temperature(&mut self, temperature: f64) -> &mut Self {
//        self.temperature = Some(temperature);
//        self
//    }
//
//    fn set_stepsize(&mut self, stepsize: f64) -> &mut Self {
//        self.stepsize = Some(stepsize);
//        self
//    }
//
//    fn build(&self) ->
//        Result<Box<impl Optimizer<T>>, IncompleteOptimizerBuild>
//    {
//        if let Some(method) = self.method.clone() {
//            let step = if let Some(stepsize) = self.stepsize.clone() {
//                stepsize
//            } else {
//                // should set to some reasonable number, for now I'm just using 0.25
//                0.25
//            };
//            if let Some(data_vec) = self.position.clone() {
//                let low = if let Some(lower) = self.lower_bound.clone() {
//                    lower
//                } else {
//                    vec![-f64::INFINITY; data_vec.len()]
//                };
//                let up = if let Some(upper) = self.upper_bound.clone() {
//                    upper
//                } else {
//                    vec![f64::INFINITY; data_vec.len()]
//                };
//                let temp = if let Some(temperature) = self.temperature.clone() {
//                    temperature
//                } else {
//                    0.0
//                };
//                if let Some(object) = self.object.clone() {
//                    let particles = if let Some(particle_number) = self.particle_number {
//                        let mut particles: Vec<Particle<T>> = Vec::new();
//                        for _ in 0..particle_number {
//                            let particle = ParticleBuilder::new()
//                                .set_data(data_vec.to_vec()).unwrap()
//                                .set_lower(low.to_vec()).unwrap()
//                                .set_upper(up.to_vec()).unwrap()
//                                .set_objective(object.clone())
//                                .set_temperature(temp).unwrap()
//                                .set_stepsize(step).unwrap()
//                                .set_method(method)
//                                .build().unwrap();
//                            particles.push(particle);
//                        }
//                        particles
//                    } else {
//                        let mut particles: Vec<Particle<T>> = Vec::new();
//                        let particle = ParticleBuilder::new()
//                            .set_data(data_vec.to_vec()).unwrap()
//                            .set_lower(low.to_vec()).unwrap()
//                            .set_upper(up.to_vec()).unwrap()
//                            .set_objective(object)
//                            .set_temperature(temp).unwrap()
//                            .set_stepsize(step).unwrap()
//                            .set_method(method)
//                            .build().unwrap();
//                        particles.push(particle);
//                        particles
//                    };
//
//                    Ok(Box::new(ParticleSwarm{}))
//                } else {
//                    Err(IncompleteOptimizerBuild::NoObjective)
//                }
//            } else {
//                Err(IncompleteOptimizerBuild::NoPosition)
//            }
//        } else {
//            Err(IncompleteOptimizerBuild::NoMethod)
//        }
//    }
//}

//pub struct ParticleBuilder<T> {
//    object: Option<T>,
//    position: Option<Vec<f64>>,
//    score: Option<f64>,
//    lower_bound: Option<Vec<f64>>,
//    upper_bound: Option<Vec<f64>>,
//    temperature: Option<f64>,
//    velocity: Option<Vec<f64>>,
//    stepsize: Option<f64>,
//    method: Option<Method>,
//    rng: Xoshiro256PlusPlus,
//}
//
//impl<T: Objective + std::marker::Send + Clone> ParticleBuilder<T> {
//    fn new() -> Self where T: Objective + Clone + std::marker::Send {
//        let rng = Xoshiro256PlusPlus::from_entropy();
//        Self{
//            object: None,
//            position: None,
//            score: None,
//            lower_bound: None,
//            upper_bound: None,
//            temperature: None,
//            velocity: None,
//            stepsize: None,
//            method: None,
//            rng: rng,
//        }
//    }
//
//    fn set_data(&mut self, data: Vec<f64>) -> Result<&mut Self, InvalidParticleBuild> {
//        if type_of(&data[0]) != "f64" {
//            Err(InvalidParticleBuild::InvalidDataType)
//        } else {
//            self.position = Some(data);
//            Ok(self)
//        }
//    }
//
//    fn set_lower(&mut self, lower: Vec<f64>) ->
//        Result<&mut Self, InvalidParticleBuild>
//    {
//        if type_of(&lower[0]) != "f64" {
//            Err(InvalidParticleBuild::InvalidDataType)
//        } else {
//            self.lower_bound = Some(lower);
//            Ok(self)
//        }
//    }
//
//    fn set_upper(&mut self, upper: Vec<f64>) ->
//        Result<&mut Self, InvalidParticleBuild>
//    {
//        if type_of(&upper[0]) != "f64" {
//            Err(InvalidParticleBuild::InvalidDataType)
//        } else {
//            self.upper_bound = Some(upper);
//            Ok(self)
//        }
//    }
//
//    fn set_objective(&mut self, objective: T) ->
//        &mut Self
//            where
//        T: Objective + std::marker::Send
//    {
//        self.object = Some(objective);
//        self
//    }
//
//    fn set_temperature(&mut self, temperature: f64) ->
//        Result<&mut Self, InvalidParticleBuild>
//    {
//        if type_of(&temperature) != "f64" {
//            Err(InvalidParticleBuild::InvalidDataType)
//        } else {
//            self.temperature = Some(temperature);
//            Ok(self)
//        }
//    }
//
//    fn set_stepsize(&mut self, stepsize: f64) ->
//        Result<&mut Self, InvalidParticleBuild>
//    {
//        if type_of(&stepsize) != "f64" {
//            Err(InvalidParticleBuild::InvalidDataType)
//        } else {
//            self.stepsize = Some(stepsize);
//            Ok(self)
//        }
//    }
//
//    fn set_method(&mut self, method: Method) -> &mut Self {
//        self.method = Some(method);
//        self
//    }
//
//    fn build(&self) -> Result<Particle<T>, IncompleteParticleBuild> {
//        if let Some(object) = self.object.clone() {
//            if let Some(position) = self.position.clone() {
//                let lower_bound = if let Some(lower) = self.lower_bound.clone() {
//                    lower
//                } else {
//                    vec![-f64::INFINITY; position.len()]
//                };
//                let upper_bound = if let Some(upper) = self.upper_bound.clone() {
//                    upper
//                } else {
//                    vec![f64::INFINITY; position.len()]
//                };
//                let temperature = if let Some(temp) = self.temperature.clone() {
//                    temp
//                } else {
//                    0.0
//                };
//                let velocity = if let Some(vel) = self.velocity.clone() {
//                    vel
//                } else {
//                    vec![0.0; position.len()]
//                };
//                if let Some(stepsize) = self.stepsize.clone() {
//                    if let Some(method) = self.method.clone() {
//                        let rng = Xoshiro256PlusPlus::from_entropy();
//                        let mut particle = Particle {
//                            object: object,
//                            position: position.clone(),
//                            prior_position: position.to_vec(),
//                            best_position: position.to_vec(),
//                            best_score: f64::INFINITY,
//                            score: f64::INFINITY,
//                            lower_bound: lower_bound,
//                            upper_bound: upper_bound,
//                            temperature: temperature,
//                            velocity: velocity.clone(),
//                            prior_velocity: velocity.to_vec(),
//                            stepsize: stepsize,
//                            rng: rng,
//                            method: method,
//                        };
//                        particle.score = particle.evaluate();
//                        particle.best_score = particle.score;
//                        Ok(particle)
//                    } else {
//                        Err(IncompleteParticleBuild::NoMethod)
//                    }
//                } else {
//                    Err(IncompleteParticleBuild::NoStepsize)
//                }
//            } else {
//                Err(IncompleteParticleBuild::NoPosition)
//            }
//        } else {
//            Err(IncompleteParticleBuild::NoObjective)
//        }
//    }
//}
//
//#[derive(Debug)]
//pub struct Particle<T> {
//    object: T,
//    position: Vec<f64>,
//    prior_position: Vec<f64>,
//    best_position: Vec<f64>,
//    best_score: f64,
//    score: f64,
//    lower_bound: Vec<f64>,
//    upper_bound: Vec<f64>,
//    temperature: f64,
//    velocity: Vec<f64>,
//    prior_velocity: Vec<f64>,
//    stepsize: f64,
//    method: Method,
//    rng: Xoshiro256PlusPlus,
//}

//impl<T: Objective + std::marker::Send> Particle<T> {
//    pub fn new(
//        data: Vec<f64>,
//        lower: Vec<f64>,
//        upper: Vec<f64>,
//        object: T,
//        temperature: f64,
//        stepsize: f64,
//        method: &Method,
//    ) -> Particle<T>
//        where
//            T: Objective + std::marker::Send,
//    {
//
//        let mut init_rng = Xoshiro256PlusPlus::from_entropy();
//        // initialize velocity for each parameter to zero
//        let mut v = vec![0.0; data.len()];
//        // adjust the starting velocity if we're doing particle swarm
//        //let mut rng = thread_rng();
//        match method {
//            Method::ParticleSwarm => {
//                v.iter_mut()
//                    .zip(&lower)
//                    .zip(&upper)
//                    .for_each(|((vel, low), up)| {
//                        // draw velocities from uniform dist from +/-(range/40)
//                        let init_range = (up - low) / 40.0;
//                        let nudge = init_rng.gen_range(-init_range..init_range);
//                        *vel = *vel + nudge;
//                });
//            },
//            Method::PIDAO => {
//                todo!()
//            }
//            _ => (),
//        }
//        // copy of velocity to place something into prior_velocity
//        let pv = v.to_vec();
//        // copy of data to place something into prior_position
//        let d = data.to_vec();
//        let pr = data.to_vec();
//
//        let rng = Xoshiro256PlusPlus::from_entropy();
//        let mut particle = Particle {
//            object: object,
//            position: data,
//            prior_position: d,
//            best_position: pr,
//            best_score: f64::INFINITY,
//            score: f64::INFINITY,
//            lower_bound: lower,
//            upper_bound: upper,
//            temperature: temperature,
//            velocity: v,
//            prior_velocity: pv,
//            stepsize: stepsize,
//            rng: rng,
//            method: *method,
//        };
//        particle.score = particle.evaluate();
//        particle.best_score = particle.score;
//        particle
//    }
//
//    /// Gets the score for this Particle
//    fn evaluate(
//            &self,
//    ) -> f64 {
//        self.object.objective(&self.position)
//    }
//
//    /// Adjusts the position of the Particle
//    /// Note that all [Particle]s are instantiated with a velocity of zero.
//    /// Therefore, if your optimization algorith does not make use of velocity,
//    /// the velocity is never adjusted away from zero, so adding it here does
//    /// nothing. If your method *does* use velocity, then it will have adjusted
//    /// the velocity such that adding it here has an effect on its position.
//    /// Complementary to that, if you want only the velocity to affect particle
//    /// position, but no random jitter, set stepsize to 0.0.
//    /// Modifies self.position in place.
//    fn perturb(&mut self) {
//
//        // before we change the position, set prior position to current position
//        // this will enable reversion to prior state if we later reject the move
//        self.prior_position = self.position.to_vec();
//
//        // which index will we be nudging?
//        let idx = self.choose_param_index();
//        // by how far will we nudge?
//        let jitter = self.get_jitter();
//
//        match self.method {
//            Method::SimulatedAnnealing => {
//                // nudge the randomly chosen index by jitter
//                self.position[idx] += jitter;
//            },
//            Method::ReplicaExchange => {
//                // nudge the randomly chosen index by jitter
//                self.position[idx] += jitter;
//            },
//            Method::BayesianOptimization => {
//                // nudge the randomly chosen index by jitter
//                self.position[idx] += jitter;
//            },
//            // don't use jitter for particleswarm or PIDAO
//            _ => (),
//        }
//        // add velocity element-wise to position
//        self.position.iter_mut() // mutably iterate over each position
//            .zip(&self.velocity) // in lockstep with velocity in each dimension
//            .zip(&self.lower_bound) // and the lower bounds for each dim
//            .zip(&self.upper_bound) // and the upper bounds for each dim
//            .for_each(|(((a, b), c), d)| {
//                 // update pos, keeping slightly inward of upper and lower bounds
//                *a = (*a + b).clamp(*c+f64::EPSILON, *d-f64::EPSILON)
//            })
//    }
//
//    /// Randomly chooses the index of position to update using jitter
//    fn choose_param_index(&mut self) -> usize {
//        let die = Uniform::new(0, self.position.len());
//        die.sample(&mut self.rng)
//    }
//
//    /// Sample once from a normal distribution with a mean of 0.0 and std dev
//    /// of self.stepsize.
//    fn get_jitter(&mut self) -> f64 {
//        // sample from normal distribution one time
//        /////////////////////////////////////////////////////
//        // Keep this self.rng.lock().unrwrap() for now. the argmin people use it
//        // They say it's necessary for thread-safe optims
//        /////////////////////////////////////////////////////
//
//        /////////////////////////////////////////////////////
//        // I keep wanting to make thist distr a field of Particle, but shouldn't:
//        // I want to be able to update stepsize as optimization proceeds
//        /////////////////////////////////////////////////////
//        let jitter_distr = Normal::new(0.0, self.stepsize).unwrap();
//        jitter_distr.sample(&mut self.rng)
//    }
//
//    /// Set the velocity of the Particle
//    fn set_velocity(&mut self, inertia: &f64,
//            local_weight: &f64, global_weight: &f64,
//            global_best_position: &Vec<f64>) {
//
//        /////////////////////////////////////////////////////////////////////
//        /////////////////////////////////////////////////////////////////////
//        /////////////////////////////////////////////////////////////////////
//        // I think that this is where I can get dx_dt in for PIDAO
//        /////////////////////////////////////////////////////////////////////
//        /////////////////////////////////////////////////////////////////////
//        /////////////////////////////////////////////////////////////////////
//        match self.method {
//            // if we are doing particle swarm, just update and move on
//            Method::ParticleSwarm => {
//         
//                // before we change the velocity,
//                // set prior velocity to current velocity
//                // this will enable reversion to prior state
//                // if we later reject the move
//                self.prior_velocity = self.velocity.to_vec();
//                // set stochastic element of weights applied to
//                // local and global best pos
//                // self.rng.gen samples from [0.0, 1.0)
//                let r_arr: [f64; 2] = self.rng.gen();
//                // set the new velocity
//                self.velocity.iter_mut() // mutably iterate over current velocity
//                    .zip(&self.best_position) // in lockstep with this Particle's
//                                              // best position
//                    .zip(global_best_position) // and the global best position
//                    .zip(&self.position) // and the current position
//                    .zip(&self.lower_bound) // and the lower bound
//                    .zip(&self.upper_bound) // and the upper bound
//                    // a=vel, b=local_best, c=swarm_best, d=pos,
//                    // e=lower_bound, f=upper_bound
//                    .for_each(|(((((a, b), c), d), e), f)| {
//                        //let range = f - e;
//                        let term1 = inertia * *a;
//                        // attraction to the particle's own best gets
//                        // stronger with distance
//                        let term2 = local_weight * r_arr[0] * (b - d);
//                        // attraction to the swarms's best gets
//                        // stronger with distance
//                        let term3 = global_weight * r_arr[1] * (c - d);
//                        // repulsion from lower bound defined by
//                        // squared distance to lower bound
//                        //let term4 = -(range / 100.0) / ((e - d) * (e - d));
//                        // repulsion from upper bound defined by squared
//                        // distance to upper bound
//                        //let term5 = -(range / 100.0) / ((f - d) * (f - d));
//                        *a = term1 + term2 + term3// + term4 + term5
//                    })
//            },
//            Method::PIDAO => {
////    def step(self, closure=None):
////        """Performs a single optimization step.
////        Arguments:
////            closure (callable, optional): A closure that reevaluates the model
////                and returns the loss.
////        """
////        loss = None
////        if closure is not None:
////            loss = closure()
////
////        for group in self.param_groups:
////            weight_decay = group['weight_decay']
////            momentum = group['momentum']
////            dampening = group['dampening']
////            nesterov = group['nesterov']
////            lr = group['lr']
////            kp = group['kp']
////            ki = group['ki']
////            kd = group['kd']
////            for p in group['params']:
////                if p.grad is None:
////                    continue
////                d_p = p.grad.data
////                if weight_decay != 0:
////                    d_p.add_(p.data, alpha=weight_decay)
////                if momentum != 0:
////                    param_state = self.state[p]
////                    if 'z_buffer' not in param_state:
////                        z_buf = param_state['z_buffer'] = torch.zeros_like(p.data)
////                        z_buf.add_(d_p, alpha=lr)
////                    else:
////                        z_buf = param_state['z_buffer']
////                        z_buf.add_(d_p, alpha=lr)
////
////                    if 'y_buffer' not in param_state:
////                        param_state['y_buffer'] = torch.zeros_like(p.data)
////                        y_buf = param_state['y_buffer']
////                        y_buf.add_(d_p, alpha=-lr*(kp - momentum * kd)).add_(z_buf, alpha=-ki * lr)
////                        y_buf.mul_((1 + momentum * lr) ** -1)
////                    else:
////                        y_buf = param_state['y_buffer']
////                        y_buf.add_(d_p, alpha=-lr * (kp - momentum * kd)).add_(z_buf, alpha=-ki * lr)
////                        y_buf.mul_((1 + momentum * lr) ** -1)
////
////                    d_p = torch.zeros_like(p.data).add_(y_buf, alpha=lr).add_(d_p, alpha=-kd*lr)
////                p.data.add_(d_p)
////
////        return loss
//                todo!()
//            },
//            _ => (),
//        }
//    }
//
//    fn step(
//            &mut self,
//            inertia: &f64,
//            kp: &f64,
//            ki: &f64,
//            kd: &f64,
//            local_weight: &f64,
//            global_weight: &f64,
//            global_best_position: &Vec<f64>,
//            t_adj: &f64,
//            accept_from_logit: &bool,
//    )
//        where
//            T: Objective + std::marker::Send,
//    {
//        // set the new velocity
//        self.set_velocity(
//            inertia,
//            local_weight,
//            global_weight,
//            global_best_position,
//            //kp,
//            //ki,
//            //id,
//        );
//        // move the particle. Takes into account stepsize for jitter in a single
//        //  dimension, and velocity over all dimensions.
//        self.perturb();
//
//        let score = self.evaluate(); //, rec_db, kmer, max_count, alpha);
//
//        match self.method {
//            // if we are doing particle swarm, just update and move on
//            Method::ParticleSwarm => self.update_scores(&score),
//            // If we are doing simulated annealing,
//            //  determine whether we accept the move.
//            //  if we reject, revert to prior state and perturb again.
//            Method::SimulatedAnnealing => {
//                if !self.accept(&score, accept_from_logit) {
//                    self.revert();
//                } else {
//                    // Update prior score [and possibly the best score] if we accepted
//                    self.update_scores(&score);
//                }
//            }
//            Method::ReplicaExchange => {
//                if !self.accept(&score, accept_from_logit) {
//                    self.revert();
//                } else {
//                    // Update prior score [and possibly the best score] if we accepted
//                    self.update_scores(&score);
//                }
//            }
//            Method::BayesianOptimization => todo!(),
//            Method::PIDAO => self.update_scores(&score)
//        }
//        // adjust the temperature downward
//        ////////////////////////////////////////////////////
//        // NOTE: this could be included in the match control flow above
//        // I have to check if temp adjustment is used in replica exchange,
//        // or whether we just let the temps remain constant.
//        ////////////////////////////////////////////////////
//        self.adjust_temp(t_adj);
//    }
//
//    /// Update score fields after accepting a move
//    fn update_scores(&mut self, score: &f64) {
//        self.score = *score;
//        // if this was our best-ever score, update best_score and best_position
//        if *score < self.best_score {
//            self.best_score = *score;
//            self.best_position = self.position.to_vec();
//        }
//    }
//    
//    /// Revert current position and velocity to prior values
//    fn revert(&mut self) {
//        self.position = self.prior_position.to_vec();
//        self.velocity = self.prior_velocity.to_vec();
//    }
//    
//    /// Determine whether to accept the new position, or to go back to prior
//    /// position and try again. If score is greater that prior score,
//    /// return true. If the score is less than prior score, determine whether
//    /// to return true probabilistically using the following function:
//    ///
//    /// `exp(-(score - prior_score)/T)`
//    ///
//    /// where T is temperature.
//    fn accept(&mut self, score: &f64, accept_from_logit: &bool) -> bool {
//        if *accept_from_logit {
//            // clamp score to just barely above -1.0, up to just barely below 0.0
//            // this avoids runtime error in logit
//            let clamp_score = score.clamp(-(1.0-f64::EPSILON), -f64::EPSILON);
//            // take opposite of scores, since they're opposite of AMI
//            let diff = logit(&-clamp_score)
//                - logit(&-self.score.clamp(-(1.0-f64::EPSILON), -f64::EPSILON));
//            // testing if diff >= 0.0 here, and NOT <= 0.0, since we're doing
//            //   that thing of when we compare the logit of the opposite of
//            //   the scores. Therefore, a greater score than our previous
//            //   score is desireable.
//            if diff >= 0.0 {
//                true
//            // if score > last score, decide probabilistically whether to accept
//            } else {
//                // this is the function used by scipy.optimize.basinhopping for P(accept)
//                // the only difference here is that, again, because we're really maximizing
//                // a score, we just use diff, and not -diff.
//                let accept_prob = (diff/self.temperature).exp();
//                if accept_prob > self.rng.gen() {
//                    true
//                }
//                else {
//                    false
//                }
//            }
//        } else {
//            // compare this score to prior score
//            let diff = score - self.score;
//            if diff <= 0.0 {
//                true
//            } else {
//                // this is the function used by scipy.optimize.basinhopping for P(accept)
//                let accept_prob = (-diff/self.temperature).exp();
//                if accept_prob > self.rng.gen() {
//                    true
//                }
//                else {
//                    false
//                }
//            }
//
//        }
//    }
//
//    /// Adjusts the temperature of the Particle
//    ///
//    /// Defined as
//    ///
//    /// ` T_{i+1} = T_i * (1.0 - t_{adj})`
//    ///
//    /// # Arguments
//    ///
//    /// * `t_adj` - fractional amount by which to decrease the temperature of
//    ///    the particle. For instance, if current temp is 1.0 and t_adj is 0.2,
//    ///    the new temperature will be 1.0 * (1.0 - 0.2) = 0.8
//    fn adjust_temp(&mut self, t_adj: &f64) {
//        self.temperature *= 1.0 - t_adj
//    }
//
//    /// Adjusts that stepsize of the Particle
//    ///
//    /// Definaed as
//    ///
//    /// ` S_{i+1} = S_i * (1.0 - s_{adj})`
//    ///
//    /// # Arguments
//    ///
//    /// * `s_adj` - fractional amount by which to decrease the stepsize of the
//    ///     particle.
//    fn adjust_stepsize(&mut self, s_adj: &f64) {
//        self.stepsize *= 1.0 - s_adj
//    }
//}
//
/// A trait to enable optimization of a set of parameters for
/// any struct
pub trait Objective {
    fn objective(&self, theta: &Vec<f64>) -> f64;
}

//pub struct Particles<T> {
//    particles: Vec<Particle<T>>,
//    global_best_position: Vec<f64>,
//    global_best_score: f64,
//}
//
//impl<T: Objective + Clone + std::marker::Send> Particles<T> {
//    /// Returns particles whose positions are sampled from
//    /// a normal distribution defined by the original start position
//    /// plus 1.5 * stepsize.
//    pub fn new(
//            n_particles: usize,
//            data: Vec<f64>,
//            lower: Vec<f64>,
//            upper: Vec<f64>,
//            temperature: f64,
//            stepsize: f64,
//            initial_jitter: f64,
//            object: T,
//            method: &Method,
//    ) -> Particles<T>
//        where
//            T: Objective + Clone,
//    {
//        // set variance of new particles around data to initial_jitter^2
//        let distr = Normal::new(0.0, initial_jitter).unwrap();
//        //let rng = Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy()));
//        let mut rng = Xoshiro256PlusPlus::from_entropy();
//        //let mut lrng = rng.lock().unwrap();
//
//        // instantiate a Vec
//        let mut particle_vec: Vec<Particle<T>> = Vec::new();
//
//        // if doing ReplicaExchange, we'll need geomspace vec of temps
//        let t_start = temperature / 3.0;
//        let log_start = t_start.log10();
//        let t_end = temperature * 3.0;
//        let log_end = t_end.log10();
//
//        let dt = (log_end - log_start) / ((n_particles - 1) as f64);
//        let mut temps = vec![log_start; n_particles];
//        for i in 1..n_particles {
//            temps[i] = temps[i - 1] + dt;
//        }
//
//        // instantiate particles around the actual data
//        for i in 0..n_particles {
//            //
//            let obj_i = object.clone();
//            // instantiate the random number generator
//            let mut data_vec = data.to_vec();
//            let mut temp = 0.0;
//            // first particle should be right on data
//            match self.method {
//                Method::SimulatedAnnealing => {
//                    temp = temperature;
//                }
//                Method::ReplicaExchange => {
//                    temp = temps[i].exp();
//                }
//                Method::ParticleSwarm => {
//                    temp = 0.0;
//                }
//                Method::BayesianOptimization => todo!(),
//                Method::PIDAO => {
//                    temp = 0.0;
//                }
//            }
//            if i == 0 {
//                // if this is the first particle, place it directly at data_vec
//                let particle = Particle::new(
//                    data_vec,
//                    lower.to_vec(),
//                    upper.to_vec(),
//                    obj_i,
//                    temp,
//                    stepsize,
//                    method,
//                );
//                particle_vec.push(particle);
//            } else {
//                let mut rng = Xoshiro256PlusPlus::from_entropy();
//                data_vec.iter_mut()
//                    .enumerate()
//                    .for_each(|(i,a)| {
//                        // set new particle's data to data + sample, clamp between bounds
//                        *a = *a + distr
//                            .sample(&mut rng)
//                            .clamp(lower[i],upper[i]);
//                    });
//                let particle = Particle::new(
//                    data_vec,
//                    lower.to_vec(),
//                    upper.to_vec(),
//                    obj_i,
//                    temp,
//                    stepsize,
//                    method,
//                );
//                particle_vec.push(particle);
//            }
//        }
//        // sort in ascending order of score
//        particle_vec.sort_unstable_by_key(|a| {
//            OrderedFloat(a.score)
//        });
//        // lowest score is best, so take first one's position and score
//        let best_pos = particle_vec[0].best_position.to_vec();
//        let best_score = particle_vec[0].best_score;
//        Particles {
//            particles: particle_vec,
//            global_best_position: best_pos,
//            global_best_score: best_score,
//        }
//    }
//
//    fn par_iter_mut(&mut self) -> rayon::slice::IterMut<Particle<T>> {
//        self.particles.par_iter_mut()
//    }
//
//    fn step(
//            &mut self,
//            inertia: &f64,
//            kp: &f64,
//            ki: &f64,
//            kd: &f64,
//            local_weight: &f64,
//            global_weight: &f64,
//            t_adj: &f64,
//            accept_from_logit: &bool,
//    ) {
//        let global_best = self.global_best_position.to_vec();
//        //for x in self.particles.iter_mut() {
//        self.par_iter_mut().for_each(|x| {
//            x.step(
//                &inertia,
//                &0.0,
//                &0.0,
//                &0.0,
//                &local_weight,
//                &global_weight,
//                &global_best,
//                t_adj,
//                accept_from_logit,
//            );
//        //}
//        });
//        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.score));
//        self.global_best_position = self.particles[0].best_position.to_vec();
//        self.global_best_score = self.particles[0].best_score;
//    }
//
//    fn exchange(&mut self, odds: bool) {
//        let mut iterator = Vec::new();
//        if odds {
//            let swap_num = self.len() / 2;
//            for i in 0..swap_num {
//                iterator.push((i*2,i*2+1));
//            }
//        } else {
//            let swap_num = self.len() / 2 - 1;
//            for i in 0..swap_num {
//                iterator.push((i*2,i*2+1));
//            }
//        }
//        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.temperature));
//        for swap_idxs in iterator {
//            let temp_i = self.particles[swap_idxs.1].temperature;
//            let temp_i_plus_one = self.particles[swap_idxs.0].temperature;
//            self.particles[swap_idxs.0].temperature = temp_i;
//            self.particles[swap_idxs.1].temperature = temp_i_plus_one;
//        }
//    }
//
//    /// Returns the number of particles in the Particles
//    pub fn len(&self) -> usize {self.particles.len()}
//}

///////////////////////////////////////////////////
// I may use this in acceptance criterion /////////
///////////////////////////////////////////////////
pub fn logit(p: &f64) -> f64 {
    (p / (1.0-p)).ln()
}

//pub fn optimize<T>(
//) -> (Vec<f64>, f64)
//    where
//        T: Objective + Clone + std::marker::Send,
//{
//}

//pub fn particle_swarm<T>(
//        params: Vec<f64>,
//        lower: Vec<f64>,
//        upper: Vec<f64>,
//        n_particles: usize,
//        inertia: f64,
//        local_weight: f64,
//        global_weight: f64,
//        initial_jitter: f64,
//        niter: usize,
//        object: T,
//        accept_from_logit: &bool,
//) -> (Vec<f64>, f64)
//    where
//        T: Objective + Clone + std::marker::Send,
//{
//    // turn off jitter, leaving only velocity to affect position
//    let step = 0.0;
//    let temp = 0.0;
//    let t_adj = 0.0;
//
//    let mut particles = Particles::new(
//        n_particles,
//        params,
//        lower,
//        upper,
//        temp,
//        step,
//        initial_jitter,
//        object,
//        &Method::ParticleSwarm,
//    );
//
//    for _ in 0..niter {
//        particles.step(
//            &inertia,
//            &0.0,
//            &0.0,
//            &0.0,
//            &local_weight,
//            &global_weight,
//            &t_adj,
//            accept_from_logit,
//        );
//    }
//    (particles.global_best_position.to_vec(), particles.global_best_score)
//}


