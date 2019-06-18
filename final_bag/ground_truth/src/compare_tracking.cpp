
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>
#include "Eigen/Eigen"

typedef std::vector<long double> ground_truth_data;
typedef std::vector<long double> test_data;

struct object_pair {
  int id1;
  Eigen::Vector3d p1;
  int id2;
  Eigen::Vector3d p2;
  int count;
};

int in_pair (int g_id, std::vector<object_pair> ops) {
  for (int idx = 0; idx < ops.size(); idx++) {
    if (g_id == ops[idx].id1)
      return idx;
  }
  return -1;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "compare_tracking");
  ros::NodeHandle nh("~");

  std::string path = ros::package::getPath("ground_truth");
  std::fstream filename(path + "/config/filename.txt");
  std::string f1, f2;
  getline(filename, f1, '\n');
  getline(filename, f2, '\n');
  std::fstream g_file(f1);
  std::fstream t_file(f2);


  std::vector<ground_truth_data> g_dataset;
  std::vector<test_data> t_dataset;
  std::vector<object_pair> ops;

  // read file
  if (g_file.good() && t_file.good()) {
    std::string line;
//    getline(g_file, line);

    while (getline(g_file, line, '\n')) {
      std::istringstream templine(line);
      std::string data;
      ground_truth_data g_data;
      while (getline(templine, data, ',')) {
        g_data.push_back(strtod(data.c_str(), NULL));
      }
      g_dataset.push_back(g_data);
    }
    std::cout << "Total " << g_dataset.size()
              << " ground truth data loaded." << std::endl;
    std::cout << "Each data has " << g_dataset.back().size()
              << " elements" << std::endl;

//    getline(t_file, line);
    while (getline(t_file, line, '\n')) {
      std::istringstream templine(line);
      std::string data;
      test_data t_data;
      while (getline(templine, data, ',')) {
        t_data.push_back(strtod(data.c_str(), NULL));
      }
      t_dataset.push_back(t_data);
    }
    std::cout << "Total " << t_dataset.size()
              << " test data loaded." << std::endl;
    std::cout << "Each data has " << t_dataset.back().size()
              << " elements" << std::endl;

    g_file.close();
    t_file.close();
  }
  else {
    std::cout << "Error: file cannot be oppened." << std::endl;
    return 0;
  }
 
  double avg_error = 0;
  for (int idx = 0; idx < g_dataset.size(); idx++) {
    Eigen::Vector3d p1(g_dataset[idx][2], g_dataset[idx][3], g_dataset[idx][4]);
    Eigen::Vector3d p2;
    bool found = false;
    int temp_id;
    for (int idx2 = 0; idx2 < t_dataset.size(); idx2++) {
      if (std::fabs(g_dataset[idx][0] - t_dataset[idx2][0]) < 0.001) {
        Eigen::Vector3d temp(t_dataset[idx2][2], t_dataset[idx2][3], t_dataset[idx2][4]);
        if (idx2 == 0) {
          p2 = temp;
          temp_id = t_dataset[idx2][1];
        }
        else if ((p1-temp).norm() < (p1-p2).norm() ) {
          p2 = temp;
          temp_id = t_dataset[idx2][1];
        }
        found = true;
        temp_id = t_dataset[idx2][1];
      }
    }
    /*
    std::cout << "p1 v.s. p2 = (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ") v.s. ("
              << p2.x() << ", " << p2.y() << ", " << p2.z() << ")" << std::endl;
              */
    if (found) {
      avg_error += (p1-p2).norm()/g_dataset.size();
//      std::cout << "ID = " << g_dataset[idx][1] << ", (" << p1.x() << ", " << p1.y() << ", " << p1.z() << ") v.s. ("
//                << "ID2 = " << temp_id << ", (" << p2.x() << ", " << p2.y() << ", " << p2.z() << ")" << std::endl;
    }
    else {
      avg_error += 10.0 / g_dataset.size();
    }
//    std::cout << std::endl;
  }
  std::cout << "avg_error = " << avg_error << std::endl;
  int count[100] = {0};
  for (int idx = 0; idx < g_dataset.size(); idx++) {
    int g_id = static_cast<int>(g_dataset[idx][1]);
    count[g_id]++;
    int loc = in_pair(g_id, ops);
    if (loc == -1) {
      object_pair p;
      p.id1 = g_id;
      p.p1 = Eigen::Vector3d(g_dataset[idx][2], g_dataset[idx][3], g_dataset[idx][4]);
      p.id2 = -1;
      p.p2 = Eigen::Vector3d(0, 0, 0);
      p.count = -1;
      ops.push_back(p);
      loc = ops.size()-1;
    }
    else {
      ops[loc].p1 = Eigen::Vector3d(g_dataset[idx][2], g_dataset[idx][3], g_dataset[idx][4]);
    }

    Eigen::Vector3d p1(ops[loc].p1);
    Eigen::Vector3d p2;
    bool found = false;
    int temp_id;
    for (int idx2 = 0; idx2 < t_dataset.size(); idx2++) {
      if (std::fabs(g_dataset[idx][0] - t_dataset[idx2][0]) < 0.001 ) {
        
        Eigen::Vector3d temp(t_dataset[idx2][2], t_dataset[idx2][3], t_dataset[idx2][4]);
        if (idx2 == 0) {
          temp_id = static_cast<int>(t_dataset[idx2][1]);
          p2 = temp;
//          found = true;
        }
        else if ((p1-temp).norm() < (p1-p2).norm() ) {
          temp_id = static_cast<int>(t_dataset[idx2][1]);
          p2 = temp;
        }
        found = true;
      }

    }
    if (found) {
      if (ops[loc].id2 != temp_id) {
//      	std::cout << ops[loc].id2 << ", " << temp_id << std::endl;
        ops[loc].count += 1;
      }
      ops[loc].id2 = temp_id;
      ops[loc].p2 = p2;
    }
    else {
      ops[loc].count += 1;
    }

  }
  int total_count = 0;
  for (int idx = 0; idx < ops.size(); idx++) {
    std::cout << "ID: " << ops[idx].id1 << ", count = " << ops[idx].count << "(" << count[idx]-1 << ")" << std::endl; 
    total_count += ops[idx].count;
  }
  std::cout << "Total change times: " << total_count << std::endl;
}
