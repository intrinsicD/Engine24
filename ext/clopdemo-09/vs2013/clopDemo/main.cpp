//=============================================================================
// CLOP Source
//-----------------------------------------------------------------------------
// Reference Implementation for
// Preiner et al., Continuous Projection for Fast L1 Reconstruction, In 
// Proceedings of ACM SIGGRAPH 2014
// www.cg.tuwien.ac.at/research/publications/2014/preiner2014clop
//-----------------------------------------------------------------------------
// (c) Reinhold Preiner, Vienna University of Technology, 2014
// All rights reserved. This code is licensed under the New BSD License:
// http://opensource.org/licenses/BSD-3-Clause
// Contact: rp@cg.tuwien.ac.at
//=============================================================================



#include "mixture.hpp"
#include "clop.hpp"
using namespace cp;

#include <iostream>
using namespace std;

#include "nanooff.hpp"


void main()
{
	string filenameIn = "pointcloud.off";
	string filenameOut = "pointcloud_projected.off";


	cout << "CLOP Demo" << endl;

	
	// 1. Initialize Point Set

	nanooff::Info offInfo(filenameIn);
	if (!offInfo.getSignatureOK())
	{
		cerr << "Can't load input point cloud '" << filenameIn << "'" << endl;
		return;
	}
	PointSet* points = new PointSet(offInfo.getVertexCount());
	nanooff::loadPointCloud(filenameIn, (float*)points->data());
	

	// 2. Compute mixture using HEM
	
	// set mixture parameters
	Mixture::Params mParams;
	mParams.globalInitRadius = 0.9f;		// global initialization kernel radius (applies only if useGlobalInitRadius == true)
	mParams.useGlobalInitRadius = true;		// use global initialization radius instead of NNDist sampling (more stable for non-uniform point sampling)
	mParams.useWeightedPotentials = true;	// if true, performs WLOP-like balancing of the initial Gaussian potentials
	mParams.alpha = 2.3f;					// multiple of cluster maximum std deviation to use for query radius (<= 2.5f recommended)
	mParams.nLevels = 4;					// number of levels to use for hierarchical clustering
	
	Mixture* M = new Mixture(points, mParams);


	// 3. Perform CLOP on M
	cout << endl;

	// set CLOP parameters
	clop::Params clopParams;
	clopParams.nIterations = 12;			// number of CLOP iterations
	clopParams.kernelRadius = 4.0f;			// basic kernel radius
	clopParams.doubleInitRadius = true;		// doubles the kernel radius for the initial iteration (recommended for more stable initialization in the presence of stronger outliers)
	clopParams.interleaveRepulsion = true;	// use interleaved repulsion from the CLOP paper (recommended)
	clopParams.repulsionRadiusFac = 0.5f;	// repulsion kernel radius factor for kernel cutoff (1 - full repulsion, 0 - no repulsion). recommended value is ~ 0.5
	clopParams.useSoftEta = true;			// uses the more gently decreasing eta = -r from Huang et al. instead of the original eta = 1/(3r³) from Lipman et al. (recommended)
	clopParams.mu = 0.4f;					// balancing factor between attraction and repulsion forces; E = attracion + µ * repulsion, µ \in [0, 0.5]
	clopParams.useDiscreteLOP = false;		// ignores Gaussian covariances for comparison purposes. applies original WLOP using the Gaussian centers for singular attraction only.
	
	
	// optional: take subset of input points for projection
	PointSet* particles = new PointSet();
	for (uint i = 0; i < points->size(); i += 2)
		particles->push_back(points->at(i));
	

	PointSet* projectedPoints = clop::project(particles, M, clopParams);
	

	// 4. write projected points
	nanooff::savePointCloud(filenameOut, (float*)projectedPoints->data(), projectedPoints->size());
}

