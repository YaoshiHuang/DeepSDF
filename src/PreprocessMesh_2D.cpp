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

// PreprocessMesh.cpp processes each *.obj file separately and in parallel.

// extern claims an external function.
extern pangolin::GlSlProgram GetShaderProgram();

void SampleFromSurface(
    std::vector<Eigen::Vector2f> blank,
    std::vector<Eigen::Vector2f> normal,
    std::vector<Eigen::Vector2i> line,
    std::vector<Eigen::Vector2f>& surfpts2d,
    std::vector<Eigen::Vector2f>& normals2d,
    int num_sample,
    bool ref = false) {
  float total_len = 0.0f;

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
    if (ref == true)
        normals2d.push_back(normal[ele_index]);
  }
}

// SampleSDFnearSurface refers to the variations of surface points. This can be inferred since a kdTree is input.
void SampleSDFNearSurface(
    KdVertexListTree& kdTree_surf,
    KdVertexListTree& kdTree_vert,
    std::vector<Eigen::Vector2f>& pts_surf,
    std::vector<Eigen::Vector2f>& pts_vert,
    std::vector<Eigen::Vector2f>& xy_surf,
    std::vector<Eigen::Vector2f>& normals,
    std::vector<Eigen::Vector2f>& xy,
    std::vector<float>& sdfs,
    int num_rand_samples,
    float variance,
    float second_variance,
    float bounding_cube_dim,
    int num_votes) {
  float stdv = sqrt(variance);

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
  std::vector<Eigen::Vector3f> xy_used;
  std::vector<Eigen::Vector3f> second_samples;

  std::random_device rd;
  std::mt19937 rng(rd());
  // This line is of no use, while vertices should be the second group of sampled surface points.
  std::uniform_int_distribution<int> vert_ind(0, pts_surf.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  // xy_surf refers to the sampled points on surfaces.
  for (unsigned int i = 0; i < xy_surf.size(); i++) {
    Eigen::Vector2f surface_p = xy_surf[i];
      
    // samp1 and samp2 should be two variations (positive or negative about the surface) of xy_surf[i].
    Eigen::Vector2f samp1 = surface_p;
    Eigen::Vector2f samp2 = surface_p;

    for (int j = 0; j < 2; j++) {
      samp1[j] += perterb_norm(rng);
      samp2[j] += perterb_second(rng);
    }

    xy.push_back(samp1);
    xy.push_back(samp2);
  }

  // bounding_cube_dim is set to 2, which rand_dist has a range of (0,1).
  // num_rand_samples refers to samples that randomly distribute in the domain. To be noted, points that exactly on the surface, in other words points whose sdf values equal to 0, are not included in the training set.
  // This indicates that the geometry is normalised to a unit sphere in the main function.
  for (int s = 0; s < (int)(num_rand_samples); s++) {
    xy.push_back(Eigen::Vector2f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  // now compute sdf for each xy sample
  // num_votes is set to 11. According to my experience, num_votes should refer to a hyperparameter in kdTree.
  // kdTree is included in nanoflann. In this case, it is a 3dTree, while we need a 2dTree.
  for (int s = 0; s < (int)xy.size(); s++) {
    Eigen::Vector2f samp = xy[s];
    std::vector<int> surf_indices(num_votes);
    std::vector<float> surf_distances(num_votes);
    std::vector<int> vert_indices(num_votes);
    std::vector<float> vert_distances(num_votes);
      
    // surf_indices, vert_indices and surf_distances, vert_distances are set to be empty while filled during knnSearch.
    kdTree_surf.knnSearch(samp.data(), num_votes, surf_indices.data(), surf_distances.data());
    kdTree_vert.knnSearch(samp.data(), num_votes, vert_indices.data(), vert_distances.data());

    float sdf;

    uint32_t surf_ind = surf_indices[0];
    uint32_t vert_ind = vert_indices[0];
    Eigen::Vector2f surf_vec = samp - pts_surf[surf_ind];
    Eigen::Vector2f vert_vec = samp - pts_vert[vert_ind];
    float surf_vec_leng = surf_vec.norm();
    float vert_vec_leng = vert_vec.norm();

    // if close to the surface, use point plane distance
    if (surf_vec_leng > vert_vec_leng) {
      sdf = vert_vec_leng;
      float d = normals[surf_ind].dot(surf_vec / surf_vec_leng);
        
      if (d < 0)
        sdf = -sdf;
        
      sdfs.push_back(sdf);
    } else {
        if (surf_vec_leng < stdv)
          sdf = fabs(normals[surf_ind].dot(surf_vec));
        else
          sdf = surf_vec_leng;
          float d = normals[surf_ind].dot(surf_vec / surf_vec_leng);
        
        if (d < 0)
          sdf = -sdf;
        
        sdfs.push_back(sdf);
    } 
  }
}

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

void writeSDFToNPY(
    std::vector<Eigen::Vector2f>& xy,
    std::vector<float>& sdfs,
    std::string filename) {
  unsigned int num_vert = xy.size();
  std::vector<float> data(num_vert * 3);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector2f v = xy[i];
    float s = sdfs[i];

    for (int j = 0; j < 2; j++)
      data[data_i++] = v[j];
    data[data_i++] = s;
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 3}, "w");
}

void writeSDFToNPZ(
    std::vector<Eigen::Vector2f>& xy,
    std::vector<float>& sdfs,
    std::string filename,
    bool print_num = false) {
  unsigned int num_vert = xy.size();
  std::vector<float> pos;
  std::vector<float> neg;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector2f v = xy[i];
    float s = sdfs[i];

    if (s > 0) {
      for (int j = 0; j < 2; j++)
        pos.push_back(v[j]);
      pos.push_back(s);
    } else {
      for (int j = 0; j < 2; j++)
        neg.push_back(v[j]);
      neg.push_back(s);
    }
  }

  cnpy::npz_save(filename, "pos", &pos[0], {(long unsigned int)(pos.size() / 3.0), 3}, "w");
  cnpy::npz_save(filename, "neg", &neg[0], {(long unsigned int)(neg.size() / 3.0), 3}, "a");
  if (print_num) {
    std::cout << "pos num: " << pos.size() / 3.0 << std::endl;
    std::cout << "neg num: " << neg.size() / 3.0 << std::endl;
  }
}

int main(int argc, char** argv) {
  std::string meshFileName;
  std::string npyFileName;
  bool test_flag = false;
    
  std::vector<Eigen::Vector2f> blank;
  std::vector<Eigen::Vector2f> normal;
  std::vector<Eigen::Vector2i> line;
  float variance = 0.005;
  float max_bound = 50.;
  int num_sample = 500000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.03f;
  float num_samp_near_surf_ratio = 47.0f / 50.0f;

  // Similar to the Arg_Parser module in Python
  CLI::App app{"PreprocessMesh"};
  app.add_option("-m", meshFileName, "FEM mesh File Name for Reading")->required();
  app.add_option("-o", npyFileName, "Save npy pc to here")->required();
  app.add_option("-s", num_sample, "Save ply pc to here");
  app.add_option("--var", variance, "Set Variance");
  app.add_option("--dist", max_bound, "Set max boundary");
  app.add_flag("-t", test_flag, "test_flag");

  CLI11_PARSE(app, argc, argv);

  if (test_flag)
    variance = 0.05;

  float second_variance = variance / 10;
  std::cout << "variance: " << variance << " second: " << second_variance << std::endl;
  if (test_flag) {
    second_variance = variance / 100;
    num_samp_near_surf_ratio = 45.0f / 50.0f;
    num_sample = 250000;
  }

  // Pre: According to preprocess.py and the source code of pangolin::Geometry.cpp, LoadGeometry will load *.obj files.
  // Pre: According to Pangolin/geometry_obj.cpp, LoadGeometry in default loads face elements. Not sure whether we could easily process line elements. But we do have some tricks to take advantage of these developed codes.
  std::cout << "Load blank shape curves from *.fem files" << std::endl;
  loadFEMfile(meshFileName, blank, normal, line);

  // Pre: linearize the object indices
  // Pre: const int total_num_indices = total_num_faces * 3;
  // Pre: ManagedImage refers to images that manage their own memory, storing a strong pointer to the memory.  
  // Pre: Seemingly this part is just to improve the efficiency of memory usage. new_ibo, currently being an empty Image, is created to save vertex_indices.
  // Pre: According to the source code of .SubImage, the operation of SubImage(0,0,3,total_num_faces) takes the indices of three vertices of all triangles.
  // Pre: In our blank optimisation case, it should be SubImage(0,0,2,total_num_lines) since we use line elements.
  // Pre: Faces are linearized and the vertex_indices saved in new_ibo are converted to the attributes of faces.
  // Pre: remove textures (just the next line)

  //get verticesRef
  int num_samp_near_surf = (int)(47 * num_sample / 50);
  std::cout << "num_samp_near_surf: " << num_samp_near_surf << std::endl;
  bool ref = true;
  std::vector<Eigen::Vector2f> verticesRef;
  std::vector<Eigen::Vector2f> normalsRef;
  SampleFromSurface(blank, normal, line, verticesRef, normalsRef, num_sample, ref);

  KdVertexList kdVerts_surf(verticesRef);
  KdVertexListTree kdTree_surf(2, kdVerts_surf);
  kdTree_surf.buildIndex();
    
  KdVertexList kdVerts_vert(blank);
  KdVErtextListTree kdTree_vert(kdVerts_vert);
  kdTree_vert.buildIndex();

  std::vector<Eigen::Vector2f> xy;
  std::vector<Eigen::Vector2f> xy_surf;
  std::vector<Eigen::Vector2f> normals_surf;
  std::vector<float> sdf;
  
  // xy_surf refers to surfpts in the function of SampleFromSurface. This should be passed by address.
  // Another group of surface points should be sampled to calculate sdf values.
  SampleFromSurface(blank, normal, line, xy_surf, normals_surf, num_samp_near_surf / 2);

  // xy, on the contrary, is created as an empty vector, which will be filled in the SampleSDFNearSurface function below.
  auto start = std::chrono::high_resolution_clock::now();
  SampleSDFNearSurface(
      kdTree_surf,
      kdTree_vert,
      verticesRef,
      blank,
      xy_surf,
      normalsRef,
      xy,
      sdf,
      num_sample - num_samp_near_surf,
      variance,
      second_variance,
      max_bound,
      11);

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
  std::cout << elapsed << std::endl;

  if (save_ply) {
    writeSDFToPLY(xy, sdf, plyFileNameOut, false, true);
  }

  std::cout << "num points sampled: " << xy.size() << std::endl;
  std::size_t save_npz = npyFileName.find("npz");
  if (save_npz == std::string::npos)
    writeSDFToNPY(xy, sdf, npyFileName);
  else {
    writeSDFToNPZ(xy, sdf, npyFileName, true);
  }

  std::cout << "ended correctly" << std::endl;
  return 0;
}