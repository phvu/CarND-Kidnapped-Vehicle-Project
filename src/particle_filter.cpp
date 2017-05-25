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

#include "particle_filter.h"

using namespace std;

/**
 * Template class for generating random numbers in N dimensional space
 * using N independent generators
 */
template<template<typename> class _Distribution, typename _Element>
class RandomGenerator {

  std::vector<_Distribution<_Element>> dists_;

public:

  RandomGenerator(int n, _Element means[], _Element stds[]) {
    for (int i = 0; i < n; ++i) {
      dists_.push_back(_Distribution<_Element>(means[i], stds[i]));
    }
  }

  _Element next(int i, default_random_engine &engine) {
    return (dists_[i])(engine);
  }
};

//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 5;

  double means[] = {x, y, theta};
  default_random_engine gen;
  RandomGenerator<normal_distribution, double> generator(3, means, std);

  particles.reserve((size_t)num_particles);
  for (int i = 0; i < num_particles; ++i) {
    Particle p = {};
    p.id = i;
    p.x = generator.next(0, gen);
    p.y = generator.next(1, gen);
    p.theta = generator.next(2, gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  bool zero_yaw_rate = std::abs(yaw_rate) <= std::numeric_limits<double>::epsilon();
  default_random_engine gen;

  for (auto &&p: particles) {
    double means[3];

    if (zero_yaw_rate) {
      double s = velocity * delta_t;
      means[2] = p.theta;
      means[0] = p.x + s * cos(means[2]);
      means[1] = p.y + s * sin(means[2]);
    } else {
      double s = velocity / yaw_rate;
      double prev_theta = p.theta;
      means[2] = prev_theta + yaw_rate * delta_t;
      means[0] = p.x + s * (sin(means[2]) - sin(prev_theta));
      means[1] = p.y + s * (cos(prev_theta) - cos(means[2]));
    }

    RandomGenerator<normal_distribution, double> generator(3, means, std_pos);
    p.x = generator.next(0, gen);
    p.y = generator.next(1, gen);
    p.theta = generator.next(2, gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // do nothing
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

  // pre-compute some constants for the multi-variate Gaussian distribution
  double twice_var_x = 2 * std_landmark[0] * std_landmark[0];
  double twice_var_y = 2 * std_landmark[1] * std_landmark[1];
  double coeff = -log(2 * M_PI * std_landmark[0] * std_landmark[1]);

  weights.clear();
  weights.reserve((size_t)num_particles);
  for (auto &&p: particles) {
    p.weight = 0;

    for (auto &&o: observations) {
      // use sensor_range to filter the list of observations
      // https://discussions.udacity.com/t/why-does-updateweights-function-needs-sensor-range/248695
      if (abs(o.x) > sensor_range or abs(o.y) > sensor_range) {
        continue;
      }
      // given the current position of p, and the observation o,
      // compute the location of the observation in the Map's coordinates
      double projected_x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
      double projected_y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);

      // nearest-neighbor
      double min_distance = std::numeric_limits<double>::max();
      double matched_x = 0, matched_y = 0;
      for (auto &&l: map_landmarks.landmark_list) {
        double d = dist(projected_x, projected_y, l.x_f, l.y_f);
        if (min_distance > d) {
          min_distance = d;
          matched_x = l.x_f;
          matched_y = l.y_f;
        }
      }

      // probability of the projected coordinates, as a multi-variate Gaussian at mean (matched_x, match_y)
      // and stddev std_landmark
      p.weight += coeff - (((projected_x - matched_x) * (projected_x - matched_x)) / twice_var_x)
                  - (((projected_y - matched_y) * (projected_y - matched_y)) / twice_var_y);
    }
    p.weight = exp(p.weight);
    weights.push_back(p.weight);
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles;
  default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  new_particles.reserve((size_t)num_particles);
  for (int i = 0; i < num_particles; ++i) {
    int idx = dist(gen);
    Particle p = particles[idx];
    new_particles.push_back(p);
  }
  std::copy(new_particles.begin(), new_particles.end(), particles.begin());
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
