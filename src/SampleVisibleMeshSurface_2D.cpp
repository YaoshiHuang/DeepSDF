// Copyright 2004-present Facebook. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

void loadFEMfile(
    std::string meshFileName,
    std::vector<Eigen::Vector2f>& blank,
    std::vector<Eigen::Vector2f>& normal,
    std::vector<Eigen::Vector2i>& line
    ) {
    ifstream fem;
    fem.open(meshFileName.data());
    assert(fem.is_open());
    
    string s;
    string s_slice;
    int counter = 0;
    float x_coord;
    float y_coord;
    Marix2f rot;
    Eigen::Vector2f vec_diff
    
    rot << 0, 1,
           -1, 0;
    
    while(getline(fem, s))
    {
        if (counter < 0)
        {
            vec_diff = blank[0] - blank[blank.size()-1];
            normal.push_back(rot * vec_diff / vec_diff.norm());
            line.push_back(Eigen::Vector2i(blank.size()-1, 0));
            break;
        } else if (counter == 0) {
            if (s.length <= 8)
            {
                continue;
            }
            
            s_slice = s.substr(s.length()-8, s.length()-5);
            
            if (s_slice != "GRID")
            {
                continue;
            }
            
            getline(fem, s);
        } else {            
            s_slice = s.substr(24, 32);
            x_coord = atof(s_slice.c_str());
            s_slice = s.substr(32, 40);
            y_coord = atof(s_slice.c_str());
            
            blank.push_back(Eigen::Vector2f(
                x_coord, y_coord));
                
            counter++;
            
            if (counter > 1) {
                vec_diff = blank[counter-1] - blank[counter-2];
                normal.push_back(rot * vec_diff / vec_diff.norm());
                line.push_back(Eigen::Vector2i(counter-2, counter-1));
            }
            
            if (s[0] == "$") {
                counter = -1;
            }
        }
    }
}

void SavePointsToNPY(
    std::vector<Eigen::Vector2f>& xy,
    std::string filename) {
  unsigned int num_vert = xy.size();
  std::vector<float> data(num_vert * 2);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector2f v = xy[i];

    for (int j = 0; j < 2; j++)
      data[data_i++] = v[j];
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 2}, "w");
}

void SampleFromSurface(
    std::vector<Eigen::Vector2f> blank,
    std::vector<Eigen::Vector2f> normal,
    std::vector<Eigen::Vector2i> line,
    std::vector<Eigen::Vector2f>& surfpts2d,
    int num_sample) {
  float total_len = 0.0f;

  for (int j = 0; j < blank.size(); j++) {
    surfpts2d.push_back(blank[j]);
  }
    
  std::vector<float> cdf_by_len;
  Eigen::Vector2f temp;

  // LineLen is included in Utils.h.
  for (const Eigen::Vector2i& ele : line) {
    float len = LineLen(
        (Eigen::Vector2f)Eigen::Map<Eigen::Vector2f>(vertices.RowPtr(ele(0))),
        (Eigen::Vector2f)Eigen::Map<Eigen::Vector2f>(vertices.RowPtr(ele(1))));

    if (std::isnan(len)) {
      len = 0.f;
    }

    total_len += len;

    if (cdf_by_len.empty()) {
      cdf_by_len.push_back(len);

    } else {
      cdf_by_len.push_back(cdf_by_len.back() + len);
    }
  }

  std::random_device seeder;
  // mt19937 creates a pseudo random number generator.
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_len);

  // This operation sample points on triangles weighted by len.
  while ((int)surfpts2d.size() < num_sample) {
    float ele_sample = rand_dist(generator);
    std::vector<float>::iterator ele_index_iter =
        lower_bound(cdf_by_len.begin(), cdf_by_len.end(), ele_sample);
      
    // The substraction of two iterators is a signed integer, which is required in next line.
    int ele_index = ele_index_iter - cdf_by_len.begin();

    const Eigen::Vector2i& ele = line[ele_index];

    // SamplePointFromLine is included in Utils.h.
    temp = SamplePointFromLine(
        Eigen::Map<Eigen::Vector2f>(blank.RowPtr(ele(0))),
        Eigen::Map<Eigen::Vector2f>(blank.RowPtr(ele(1)))
    );
      
    surfpts2d.push_back(temp);
  }
}

int main(int argc, char** argv) {
  std::string meshFileName;
  std::string npyOutFile;
  // std::string normalizationOutputFile;
  int num_sample = 6000;

  CLI::App app{"SampleVisibleMeshSurface"};
  app.add_option("-m", meshFileName, "*fem mesh File Name for Reading")->required();
  app.add_option("-o", npyOutFile, "Save npy pc to here")->required();
  // app.add_option("-n", normalizationOutputFile, "Save normalization");
  app.add_option("-s", num_sample, "number of samples on the blank");

  CLI11_PARSE(app, argc, argv);

  std::cout << "Load blank shape curves from *.fem files" << std::endl;
  loadFEMfile(meshFileName, blank, normal, line);

  std::vector<Eigen::Vector2f> xy_surf;
  SampleFromSurface(blank, normal, line, xy_surf, num_sample - blank.size());
  SavePointsToNPY(xy_surf, npyOutFile);

  // if (!normalizationOutputFile.empty()) {
  //   const std::pair<Eigen::Vector3f, float> normalizationParams =
  //       ComputeNormalizationParameters(geom);

  //   SaveNormalizationParamsToNPZ(
  //       normalizationParams.first, normalizationParams.second, normalizationOutputFile);
  // }

  std::cout << "ended correctly" << std::endl;
  return 0;
}