/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "Eigen/Dense"

#include "particle_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    
    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std[0]);
    
    // TODO: Create normal distributions for y and psi
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_psi(theta, std[2]);
    
    default_random_engine gen;
    
    
    
    for (int i = 0; i < num_particles; ++i) {
        // TODO: Sample  and from these normal distrubtions like this:
        //	 sample_x = dist_x(gen);
        //	 where "gen" is the random engine initialized earlier.

        Particle particle;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_psi(gen);
        particle.id = i;
        particle.weight = 1;
        
        particles.push_back(particle);
        weights.push_back(1);
    
        // Print your samples to the terminal.
        cout << "Sample " << i + 1 << " " << particle.x << " " << particle.y << " " << particle.theta << endl;
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    std::random_device rd;
    std::default_random_engine gen(rd());
    
    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];
        
        //update prediction for each particle
        if(fabs(yaw_rate) < 1e-6){
            particle.x = particle.x + (velocity * delta_t * cos(particle.theta));
            
            particle.y = particle.y + (velocity * delta_t * sin(particle.theta));;
            
            particle.theta = particle.theta + yaw_rate * delta_t;
        }
        else{
            particle.x = particle.x + (velocity/yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            
            particle.y = particle.y + (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            
            particle.theta = particle.theta + yaw_rate * delta_t;
        }
        
        
        //add gaussian noise
        normal_distribution<double> dist_x(particle.x, std_pos[0]);
        normal_distribution<double> dist_y(particle.y, std_pos[1]);
        normal_distribution<double> dist_psi(particle.theta, std_pos[2]);
        
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_psi(gen);
        
        particles[i] = particle;
        
        
        
        // Print your samples to the terminal.
//        cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_psi << endl;
    }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    double distance, min_distance, px, py;
    for (int i = 0; i < observations.size(); ++i) {
        min_distance = 1e20;
        px = observations[i].x;
        py = observations[i].y;
        // find out the closest measurement over landmarks
        for (int j = 0; j < predicted.size(); ++j) {
            distance = dist(px, py, predicted[j].x, predicted[j].y);
            if (distance <= min_distance) {
                min_distance = distance;
                observations[i].id = j;
            }
        }
    }
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    
    LandmarkObs obs;
    vector<LandmarkObs> landmarks_range;
    
    
    //sigma
    Eigen::MatrixXd sigma = Eigen::MatrixXd(2, 2);
    sigma << std_landmark[0] * std_landmark[0], 0.0, 0.0, std_landmark[1] * std_landmark[1];
    
    Eigen::MatrixXd sigma_inv = sigma.inverse();
    double sigma_mod = sigma.determinant();
    
    
    vector<LandmarkObs> observations_transformed;
    
    double weight_sum = 0;
    
    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];
        landmarks_range.clear();
        // Get the landmarks in range, save in observation of map coordinate
        for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
            if (dist(particle.x, particle.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f)
                <= sensor_range) {
                obs.x = map_landmarks.landmark_list[i].x_f;
                obs.y = map_landmarks.landmark_list[i].y_f;
                landmarks_range.push_back(obs);
            }
        }
        
        
        
        
        //transform observations from vehcle coordinate to map coordinate
        observations_transformed.clear();
        double x,y;
        for (int j = 0; j < observations.size(); j++){
            obs = observations[j];
            x = obs.x;
            y = obs.y;
            obs.x = particle.x + x * cos(particle.theta) - y * sin(particle.theta);
            obs.y = particle.y + x * sin(particle.theta) + y * cos(particle.theta);
            
            observations_transformed.push_back(obs);
        }
        
        dataAssociation(landmarks_range, observations_transformed);
        
        //calculate weights
        for (int j = 0; j < observations_transformed.size(); j++) {
            obs = observations_transformed[j];
            int landmark_id = obs.id;
            LandmarkObs nearest_land_mark = landmarks_range[landmark_id];
            Eigen::VectorXd x_i = Eigen::VectorXd(2);
            Eigen::VectorXd mu_i = Eigen::VectorXd(2);
            x_i << obs.x, obs.y;
            mu_i << nearest_land_mark.x, nearest_land_mark.y;
            Eigen::VectorXd diff = x_i - mu_i;
            
//            cout << "asso " << obs.x << " " << obs.y << " " << nearest_land_mark.x << " " << nearest_land_mark.y << endl;
            
            //mult-variate Gaussian distribution
            double p = exp(-0.5 * diff.transpose() * sigma_inv * diff) / sqrt(2 * M_PI * sigma_mod);
//            cout << "p " << p << endl;
            // compute the probability of landmarks observation in the gaussian distribution with mean at true landmarks
            
            // update weights
            particle.weight *= p;
            // normalize weights?
        }
        particles[i] = particle;
        weights[i] = particle.weight;
        weight_sum += particle.weight;
    }
    
    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];
        particle.weight = particle.weight/weight_sum;
        weights[i] = particle.weight;
        particles[i] = particle;
        
        
    }
//    cout << "weight sum" << weight_sum << endl;
//    cout << "finish update_weight" << endl;


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    for (int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[d(gen)]);
    }
    
    for (int i = 0; i < num_particles; ++i) {
        particles[i] = new_particles[i];
    }

//    // how to deal with weights now?
//    for (int i = 0; i < num_particles; i++) {
//        // weights[i] = particles[i].weight;
//        weights[i] = 1;
//        particles[i].weight = 1;
//    }
    new_particles.clear();

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
