#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <ext/algorithm> 
#include <iterator>
#include <cmath>
#include <fstream> 
#include <random>
#include <chrono>
#include <climits>
#include <limits>

////
// (C) Dirk Nuyens, KU Leuven, 2014,2015,2016,...
// (C) Dirk Nuyens, KU Leuven, 2016,...

#include "digitalseq_b2g.hpp"
#include<bits/stdc++.h>
using namespace std;

/// \brief Parse a type T from a stringstream s into a.
template <typename T>
std::stringstream& parse(std::stringstream &s, T &a)
{
    s >> a;
    return s;
}

/// \brief Parse a type T from a C string into a.
template <typename T>
std::stringstream& parse(const char *c, T &a)
{
    std::stringstream s(c);
    return parse(s, a);
}

/// \brief Print a vector to a stream, delemited by spaces.
template <typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v)
{
    for(const auto& e : v) s << e << ' ';
    return s;
}

extern "C" void shiftgen(int s, int M)
{
  typedef long double float_t;
  // obtain a seed from the system clock:
    unsigned seed = static_cast<int> (std::chrono::system_clock::now().time_since_epoch().count());
    //  std::mt19937 rng(seed); // Mersenne Twister        
    std::mt19937 rng(1); // Mersenne Twister 
  std::uniform_real_distribution<float_t> dis(0, 1);
  std::vector<float_t> shifts(s*M);
  for(int i = 0; i < s*M; ++i) shifts[i] = dis(rng);
  
  std::ofstream outf("shifts.txt");
  for(int i = 0; i < s*M; ++i)
    {
      outf << std::setprecision(16) << shifts[i] << std::endl;
      //      std::cout << std::setprecision(16) << shifts[i] << std::endl;
    }

  outf.close();

}
extern "C" void digitalshift(int s, int M, int mmin, int mmax, int nprev, int n,int r_beg,int r_end)
{
  typedef std::uint64_t uint_t;
  typedef long double float_t;
  
  // reading in the shifts we will apply
  std::ifstream is("shifts.txt");                               
  std::vector<float_t> shifts;
  float_t value;
  while (is >> value ) {
    shifts.push_back(value);
  }
  is.close();

  int m = mmax;
  typedef qmc::digitalseq_b2g<float_t, uint_t> generator_t;

  std::size_t m_ = 0, s_ = 0;
  std::uint64_t k_test = 0;

  std::vector<uint_t> Cs;
 
  //getting the generator file from point_gen.txt                           
  std::ifstream inFile0;
  inFile0.open("point_gen.txt");
  std::string line0;
  getline(inFile0,line0);
  inFile0.close();

  //putting the string name of the generator file in a stream               
  std::string line ;
  std::ifstream inFile;
  inFile.open(line0);

  //  std::ifstream sone("sobol_alpha2_Bs64.col");
  // std::string line;

  while(std::getline(inFile, line) && (s == 0 || s_ < s)) {        
    s_++;
    std::istringstream str(line);
    if(s_ == 1) {
      // read first line completely                                  
      std::copy(std::istream_iterator<uint_t>(str), std::istream_iterator <uint_t>(), std::back_inserter(Cs));
      m_ = Cs.size();
      if(m == 0) m = m_;
      assert(m <= m_);
      // now throw away those columns we don't need                    
      Cs.resize(m);
    } else {
      std::copy_n(std::istream_iterator<uint_t>(str), m, std::back_inserter(Cs));
    }
  }
  
  inFile.close();

  generator_t latgen(s, m, Cs.begin());
  float_t x_s[s];
  float_t Q_vec[mmax-mmin];
  float_t varQ_vec[mmax-mmin];
  float_t x_temp;
  float_t Y[(int)pow(2,mmax)][s];
  int N = 1 << m; // estimates with 2^m function values

  for(int r=r_beg; r<= r_end;r++)
    {
      latgen.reset();
      for(int i = 0; i < nprev;i++)
	{
	  ++latgen;
	}
	    
      int counter = 0;
      for(int k = nprev; k < n; ++k, ++latgen) {
	std::vector<float_t> x(*latgen); // get QMC point	
	for(int j = 0; j < s; ++j){
	  x_s[j] = (float_t)((uint64_t)((pow((uint64_t)2,64))*x[j])^(uint64_t)((pow((uint64_t)2,64))*shifts[(r-1)*s + j]))/(float_t)pow((float_t)2,64);      
	  Y[counter][j] = x_s[j];
	}

	counter = counter + 1;
      }
      if(r < M)
	{
	  latgen.reset();
	  for(int i = 0; i < nprev;i++)
	    {
	      ++latgen;
	    }
	}
      
      std::ofstream ouf3("shifted_points.txt");
      
      for(int lin=0;lin < counter;lin++)
	{
	  for(int col=0; col < s;col++)
	    {
	      ouf3 << std::setprecision(16) << Y[lin][col] << " " ;
	    }
	  ouf3 << std::endl ;
	}

      ouf3.close(); 

    }

  return;
}

