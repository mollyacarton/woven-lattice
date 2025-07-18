#Unit cell for woven lattice 
#Molly Carton 2022 11 03

#Woven lattice module
#

import numpy as np
import scipy.spatial
import scipy.interpolate
import matplotlib.pyplot as plt
import math
from json import dumps
from itertools import combinations,  product 


import random
# import plotly.graph_objects as go
import sys




def linspace(start, stop, n):
	"""
	MATLAB-like linspace generator (start,stop,n); inclusive
	from WOVEN 3-HELIX STRUT
	Coder: Carlos M. Portela
	California Institute of Technology
	3/15/18
	"""

	if n == 1:
		yield stop
		return
	h = (stop - start) / (n - 1)
	for i in range(n):
		yield start + h * i

def angleBetweenVectors(v1,v2):
	"""
	Minimal angle between two N-vectors with dot product
	"""
	return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


def signedAngleBetweenVectors(v1,v2,normal):
	"""
	Angle between coplanar 3-vectors, signed w/r/t normal vector
	For use in ordering helix strands.
	"""
	vn=normalize(normal)
	ans=np.arctan2(np.dot(np.cross(v1,v2),vn),np.dot(v1,v2))
	return (ans)

def sphericalToUnitCartesian(points):
	"""
	convert from <theta,phi> to unit <x,y,z>
	theta = polar angle, [-pi/2 pi/2]
	phi = azimuthal angle, [0, 2*pi]
	"""

	cartPoints = np.zeros([points.shape[0],3])
	for n in range(points.shape[0]):
		pt = points[n,:]
		incl = np.cos(pt[0])
		x = incl*np.cos(pt[1])
		y = incl*np.sin(pt[1])
		z = np.sin(pt[0])
		cartPoints[n,:] = [x,y,z]
		cartPoints[n,:] = cartPoints[n,:]/np.linalg.norm(cartPoints[n,:])
	return(cartPoints)

def sphericalToCartesian(points):
	"""
	convert from <r,theta,phi> to unit <x,y,z>
	r = radius
	theta = polar angle from equator, [-pi/2 pi/2]
	phi = azimuthal angle, [0, 2*pi]
	"""
	cartPoints = np.zeros([points.shape[0],3])
	for n in range(points.shape[0]):
		pt = points[n,:]
		incl = np.cos(pt[1])
		x = pt[0]*incl*np.cos(pt[2])
		y = pt[0]*incl*np.sin(pt[2])
		z = pt[0]*np.sin(pt[1])
		cartPoints[n,:] = [x,y,z]
	return(cartPoints)

def cartesianToSpherical(points):
	"""
	convert from <x,y,z> to <r,theta,phi>
	theta = polar angle from equator, [-pi/2 pi/2]
	phi = azimuthal angle, [0, 2*pi]
	"""
	if points.ndim>1:
		sphericalPoints = np.zeros([points.shape[0],3])
		for n in range(points.shape[0]):
			pt = points[n,:]
			radius = np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2) #np.linalg.norm(pt)
			theta = np.arcsin(pt[2]/radius)
			phi = np.arctan2(pt[1],pt[0])
			sphericalPoints[n,:] = [radius,theta,phi]
	else:
		radius = np.linalg.norm(points)
		theta = np.arcsin(points[2]/radius)
		phi = np.arctan2(points[1],points[0])
		sphericalPoints = [radius,theta,phi]
	return(sphericalPoints)


def axisAngleRotate(vector,axisVector,angle):
	"""
	rotate a vector by an angle about the axis defined by axisVector
	returns the rotated vector
	"""
	axisVector = axisVector/np.linalg.norm(axisVector)
	rot = scipy.spatial.transform.Rotation.from_rotvec(angle*axisVector) #norm gives rotation angle
	rotatedVector = rot.apply(vector)
	return(rotatedVector)

def screwTransform(vector, rotVector, invert=0):
	"""
	screw transformation from a vector ([x,y,z],theta)
	"""
	vector = np.array(vector)
	transVector = np.array(rotVector[0])
	angle = rotVector[1]

	rot = scipy.spatial.transform.Rotation.from_rotvec(angle*(transVector/np.linalg.norm(transVector))) 
	if invert:
		screwedVector  = rot.apply(vector) - transVector
		# print('-')
	else:
		screwedVector  = rot.apply(vector) + transVector
		# print('+')
	return(screwedVector)	



def polarArray(points,axisVector,numberOfRepeats):
	"""
	Polar array of points about axisVector, evenly distributed a number of times by numberOfRepeats
	"""
	arrayPts = np.zeros((numberOfRepeats*points.shape[0],points.shape[1]))
	angle = 2*np.pi/float(numberOfRepeats)
	rot = rotationMatrixAboutVector(axisVector,angle)
	for n in range(numberOfRepeats):
		arrayPts[n*points.shape[0]:(n+1)*points.shape[0],:] = np.dot(np.linalg.matrix_power(rot,n),points.T).T
	return(arrayPts)



def rotationMatrixAboutVector(axisVector,angle):
	"""
	rotate a vector by an angle about the axis defined by axisVector
	returns the rotation matrix
	"""
	axisVector = axisVector/np.linalg.norm(axisVector)
	rot = scipy.spatial.transform.Rotation.from_rotvec(angle*axisVector)
	return(np.asarray(rot.as_matrix()))

def rotationMatrixVectorToVector(vector,targetVector):
	"""
	returns rotation matrix for rotating points from relative positions about vector to about targetVector
	shortest rotation; does not maintain any particular orientation
	will have to rotate about targetVector to orient result
	"""
	axisVector = np.cross(vector,targetVector)
	dotProduct = np.dot(vector,targetVector)

	if np.linalg.norm(axisVector) < 1e-10:
		if np.sign(dotProduct) == 1:
			return(np.identity(3))
		else:
			return([[-1,0,0],[0,1,0],[0,0,-1]])
		# return(np.sign(dotProduct)*np.identity(3))
	axisVector = axisVector/np.linalg.norm(axisVector)
	angle = np.arccos(dotProduct/ (np.linalg.norm(vector)*np.linalg.norm(targetVector)) )	
	rot = scipy.spatial.transform.Rotation.from_rotvec(angle*axisVector)
	return(np.asarray(rot.as_matrix()))


def findNeighbors(points,jointRadius):
	"""
	#Find nearest neighbor point pairs using convex hull as delauney triangulation
	"""

	try:
		hull = scipy.spatial.ConvexHull(points)
	except Exception as e:
		print('Qhull encountered an error: ', repr(e))
		# #TODO: appropriate error; handle coplanar points separately
	unique_facet_equations, uniqueFacetIndices = np.unique(hull.equations, axis=0, return_inverse=1)
	mergedFacetPairs = [[] for _ in range(unique_facet_equations.shape[0])]
	for n in range(unique_facet_equations.shape[0]):
		# coplanarFacets = hull.simplices[uniqueFacetIndices==n]
		for facet in hull.simplices[uniqueFacetIndices==n]:
			mergedFacetPairs[n] += list(combinations(facet,2))
		values, inverse, counts = np.unique(mergedFacetPairs[n], axis=0, return_inverse=1, return_counts=1)
		mergedFacetPairs[n] = values[np.where(counts==1)[0]]
	pairs = np.concatenate(mergedFacetPairs, axis=0)
	pairs = np.unique(np.sort(np.array(pairs)),axis=0)
	return(pairs)


def sortNeighbors(numPts,pairs):
	"""
	Convert from edgewise into nodewise
	"""
	neighborListByPoint = []
	pairsListByPoint = []
	pairs = list(enumerate(pairs))
	for n in range(numPts):
		matches = list([idx,pair[1]] for idx, pair in pairs if pair[0]==n) + list([idx,pair[0]] for idx, pair in pairs if pair[1]==n)
		matches.sort()
		pairsListByPoint.append([match[0] for match in matches])
		neighborListByPoint.append([match[1] for match in matches])

	return(neighborListByPoint,pairsListByPoint)



def findCenterPt(pair,ptRadii):
	"""
	Find the approximate weighted center between two circles of nonequal radius on a unit sphere
	"""
	pairNorm = pair

	#TODO: fix error causing break in tetrakai
	cen = ptRadii[0]/(ptRadii[0]+ptRadii[1]) #this is an approximation, but a decent one (to ~half a degree)
	# cen = np.arcsin(np.sin(1)*np.sin(ptRadii[0])/np.sin(ptRadii[1]))/(ptRadii[0]+ptRadii[1]) ##########
	# print(cen)
	res = scipy.spatial.geometric_slerp(pairNorm[0], pairNorm[1], [0,cen,1])
	res[1] = res[1]/np.linalg.norm(res[1])
	return(res[1])


def rightSphericalTriangle(a,c):
	"""
	calculate distance and angle of tangent line between a point and a circle on a unit sphere
	# print(' helix rad = ', a,' angle = ',c)
	returns: radius of circle, tangent distance, distance to center of circle,...
	# angle from center of circle to tangent, angle from point to tangent, right angle
	"""

	C = np.pi/2
	try:
		B = np.arccos(np.tan(a)/np.tan(c))
	except:
		if np.tan(a)/np.tan(c) > 1:
			B = 0
		else:
			# raise ValueError('Impossible joint:'+ 'a and c are' + str(a)+ '' + str(c))
			error_message = {'error': True, 'message': f'Node error in right spherical triangles: a and c are {a}, {c}'}
			print(dumps(error_message), file=sys.stderr)
			quit()
	#TODO: fix the sphere checking that causes this error
	try:
		A = np.arcsin(np.sin(a)/np.sin(c))
		# b = np.arcsin(np.sin(B)*np.sin(c))
		b = np.arccos(np.cos(c)/np.cos(a)) 
	except:
		A = np.arcsin(1)
		b = np.arccos(1)
	return(np.array([a,b,c,A,B,C]))  





def findCenterTangent(centerPtNorm,helixPtNorm,helixRadiusNorm):
	"""
	#take helix point (intersection of helix centerline with joint sphere)
	#and center point between that point and its neighboring point
	#and find the line between the center point and the circle about the helix point
	#that is tangent to the circle, right-handed
	#points must be on the unit sphere
	"""

	if (np.linalg.norm(centerPtNorm)-1)>1e-4 or (np.linalg.norm(helixPtNorm)-1)>1e-4:
		print('points must be on the unit sphere: ' + str(np.linalg.norm(centerPtNorm)) + ' ' + str(np.linalg.norm(helixPtNorm)))
		centerPtNorm = centerPtNorm/np.linalg.norm(centerPtNorm)
		helixPtNorm = helixPtNorm/np.linalg.norm(helixPtNorm)

	helixRadiusOnSphere = np.arcsin(helixRadiusNorm) #go from straight path to arc length 2*np.arcsin(helixRadiusNorm/2)
	angle = angleBetweenVectors(centerPtNorm,helixPtNorm) #np.arccos(np.dot(centerPtNorm,helixPtNorm)) #path length = angle*r = subtended angle on a unit sphere
	rightTri = rightSphericalTriangle(helixRadiusOnSphere,angle)#(helixRadiusNorm,angle)
	tangentOffAngle = rightTri[3]
	tangentAngle = rightTri[4] #= angle 'B': angle from centerline to tangent point about helix center #CORRECTED - 4 not 3, B not A!
	tangentArcLength = rightTri[1] # = angle in radians on a unit sphere
	tangentChordLength = 2*np.sin(tangentArcLength/2) # chord length = 2*r*sin(subtended angle/2)
	tangentVec = centerPtNorm-helixPtNorm
	tangentLocation = scipy.spatial.geometric_slerp(helixPtNorm,centerPtNorm,helixRadiusOnSphere/angle)
	tangentLocation = axisAngleRotate(tangentLocation,helixPtNorm,tangentAngle) #and rotate it about the helix axis
	tangentLocation = tangentLocation/np.linalg.norm(tangentLocation) #normalize to sphere surface
	return tangentAngle,tangentOffAngle, tangentArcLength,np.array(tangentLocation)

def checkHelixOverlap(points,ptRadii):
	"""
	Check overlap of helix base circles for node sphere sizing
	"""

	theta = angleBetweenVectors(points[0,:],points[1,:])
	arcRadiusSum = ptRadii[0]+ptRadii[1] #np.arcsin(ptRadii[0])+np.arcsin(ptRadii[1]) # = sum of arclengths of the two helix circles
	overlap = arcRadiusSum - theta
	threshold = 1e-0
	return(overlap > -threshold, overlap)



def makeBaseHelix(helixRadius,strutLength,helixAngle,numberOfPoints,zOffset):
	"""
	# from WOVEN 3-HELIX STRUT
	# Coder: Carlos M. Portela
	# California Institute of Technology
	# 3/15/18
    Generate helix of numberOfPoints with radius helixRadius
    """

	helixCoords = np.zeros((numberOfPoints+1,3))
	for pointIndex in range(numberOfPoints+1): #from 0 to numberOfPoints
		x = helixRadius * np.cos(pointIndex*(helixAngle / numberOfPoints))
		y = helixRadius * np.sin(pointIndex*(helixAngle / numberOfPoints))
		z = zOffset + (pointIndex * strutLength / numberOfPoints)
		helixCoords[pointIndex,:]= [x,y,z]
	return(helixCoords)



####################################################################################################################
# Lattice functions
####################################################################################################################







def wovenUnitCell(unitPts,bars,sphereRadius,ptRadii,strutLengths,revolutions,helixPtNumber,curvePtNumber):#helixAngles,,backupIntoSphere):
	"""
	I: for each node, find connectivity around the node and produce tangent path/helix angles
	"""

	numPts = unitPts.shape[0]
	pairs = findNeighbors(unitPts,sphereRadius)
	numPairs = pairs.shape[0]
	unitRadii = np.array(ptRadii)/sphereRadius


	connectivity = np.zeros(numPts,dtype='int')
	for ind in range(numPts):
		connectivity[ind] = np.count_nonzero(pairs==ind)

	centers = np.zeros((numPairs,3))
	edgeListByPoint = [[] for _ in range(numPts)] #list of edges by point
	neighborTangentList = [[] for _ in range(numPts)] #list of neighbor tangent locations (endpoints of tangent path) by point
	neighborTangentAngleList = [[] for _ in range(numPts)] #list of angles of tangent point vs. edge intersection with helix circle by  point
	neighborCenterList = [[] for _ in range(numPts)] #list of center points by point
	neighborListByPoint = [[] for _ in range(numPts)]
	barListByPoint = [[] for _ in range(numPts)] #list of neighbor points by point [this makes shit all sense]
	# starts with relative point radii; increases sphere radius to keep them from overlapping

	if any(sphereRadius<sphereRadius*unitRadii):
		sphereRadius0 = float(sphereRadius)
		sphereRadius=np.amax(sphereRadius*unitRadii)+.01
		unitRadii = unitRadii*(sphereRadius0/sphereRadius)
	for n in range(numPairs):
		sphereRadius0 = float(sphereRadius)
		isOverlapping,overlap = checkHelixOverlap(unitPts[pairs[n,:]],unitRadii[pairs[n,:]])
		if isOverlapping:
			overlap = 0 if overlap < 0 else overlap
			sphereRadius1 = sphereRadius
			sphereRadius += (np.abs(overlap)*sphereRadius)/(2*np.pi) + sphereRadius*.05#(2*np.pi) + .01
			unitRadii = unitRadii*(sphereRadius1/sphereRadius)
			# strutLength = strutLength - (sphereRadius-sphereRadius0) #subtract from strut length to compensate for larger center



	strutLengths = list(map(lambda x: (x-sphereRadius)/sphereRadius, strutLengths))#strutLengths[latticePtIdx])) #keep this


	# for each pair of neighboring points, find weighted center, tangent arc between helix and center
	# and structure that data in terms of edges instead of nodes for later convenience
	for n in range(numPairs): 
		start,end = pairs[n,:] #unit point indices on this node for this edge
		startBar, endBar = bars[start],bars[end]


		centers[n,:] = findCenterPt(unitPts[pairs[n,:]],unitRadii[pairs[n,:]])
		tangentAngle1, tangentOffAngle1, tangentArcLength1, tangentLocation1 = findCenterTangent(centers[n,:],unitPts[start],unitRadii[start])
		tangentAngle2, tangentOffAngle2, tangentArcLength2, tangentLocation2 = findCenterTangent(centers[n,:],unitPts[end],unitRadii[end])

		edgeListByPoint[start].append(n) 
		edgeListByPoint[end].append(n) #list of pairs that each point appears in
		barListByPoint[start].append(endBar) #neighboring bar indices by unit point index
		barListByPoint[end].append(startBar)

		neighborTangentList[start].append(tangentLocation1) #tangent position by unit point index
		neighborTangentList[end].append(tangentLocation2) 
		neighborTangentAngleList[start].append(tangentAngle1) #tangent angle by unit point index
		neighborTangentAngleList[end].append(tangentAngle2)
		neighborCenterList[start].append(centers[n,:]) #centerpt position by unit point index
		neighborCenterList[end].append(centers[n,:])
		neighborListByPoint[start].append(end) #neighboring unit point indices by unit point index
		neighborListByPoint[end].append(start)

		#TODO: ## future. adjust to even out strands ##
		evennessRatio = .8 #how strongly to adjust them

	# neighborListByPoint,edgeListByPoint = sortNeighbors(numPts,pairs)
	return(sphereRadius,neighborTangentList,neighborTangentAngleList,neighborCenterList,
		edgeListByPoint,barListByPoint,neighborListByPoint,pairs,strutLengths)#,unitRadii)


def normalize(vector):
	"""
	Normalize vector
	"""
	normVec = vector/np.linalg.norm(vector)
	return normVec

def makeRotatedHelices(bar,point,sampling,node1,node2):#numberOfPoints,node1,node2):
	"""
	II: for each bar, make the helix according to the node tangents

	bar: the Bar instance being helixed
		bar.start: the center point of the joint at the arbitrary start of the bar
		bar.end: the center point of the joint at the arbitrary end of the bar
		bar.barRadius: the radius of the helix on the bar
		bar.neighbors1: the points at the base of the helix on the node bar.start (wrong!)
		bar.neighbors2: the points at the end of the helix on the node bar.end
	point: the unit vector describing where the center of the helix sits on the unit sphere around the point bar.start
		helix base point = bar.start + point*(sphereRadius1**2 - helixRadius**2)
	numberOfPoints: the number of points to evaluate along each helix strand
	sphereRadius1: the radius of the sphere at bar.start
	sphereRadius2: the radius of the sphere at bar.end
	"""
	sphereRadius1 = node1.sphereRadius
	sphereRadius2 = node2.sphereRadius

	idxAtNode1,idxAtNode2 = node1.bars.index(bar.idx),node2.bars.index(bar.idx)
	helixPoints1 = np.array(node1.neighborTangentList[idxAtNode1]) #np.array(bar.neighbors1)
	helixPoints2 = np.array(node2.neighborTangentList[idxAtNode2]) #np.array(bar.neighbors2)
	# helixAngles1 = np.array(node1.neighborTangentAngleList[idxAtNode1])
	# helixAngles2 = np.array(node2.neighborTangentAngleList[idxAtNode2])

	connectivity = helixPoints1.shape[0] ###TODO: NO!
	if connectivity == 0: return([],[])

	helixRadius = bar.barRadius
	revolutions = bar.barRevolutions

	#now find actual position of end of helix (slightly inside sphere), and double check that it's sensible:
	#bring helix points to coplanar to compare:
	helixPointsComp1 = helixPoints1 - point*np.sqrt(1-(helixRadius/sphereRadius1)**2)
	helixPointsComp2 = helixPoints2 + point*np.sqrt(1-(helixRadius/sphereRadius2)**2)
	crosscheck = np.absolute([np.cross(val,helixPointsComp1[n])/np.linalg.norm(np.cross(val,helixPointsComp1[n]))
							  for n,val in enumerate(helixPointsComp2)])
	if not (np.allclose(crosscheck[0],crosscheck[1])
			and np.allclose(crosscheck[1],crosscheck[2])
			and np.allclose(crosscheck[0],crosscheck[2])):
		raise ValueError('Helix ends are not parallel planes.')





	try:
		(wholerevs,shift) = divmod(bar.barRevolutions*connectivity,connectivity) #number of strands shifted by
		wholerevs,shift = int(wholerevs),int(round(shift)) #TODO: this is correct (1,1 for 4/3, 1,2 for 5/3)
	except:
		print(connectivity)
		print(node1.edgeListByPoint[idxAtNode1])
		print(node2.unitPts)
		raise ValueError('Connectivity error.')


	#sorting helixpoints1:
	angles = [signedAngleBetweenVectors(val,helixPointsComp1[0],point) for n,val in enumerate(helixPointsComp1)]
	angles = [a%(2*np.pi) for a in angles]
	helixPoints1order = np.argsort(angles)
	helixPointsComp1=helixPointsComp1[helixPoints1order]

	#sorting helixPoints2:
	#TODO: make sure this fixes the extra rev error. it does not
	comp=[signedAngleBetweenVectors(val,helixPointsComp1[0],point)%(2*np.pi) for val in helixPointsComp2]
	complh=[signedAngleBetweenVectors(helixPointsComp1[0],val,point)%(2*np.pi) for val in helixPointsComp2]
	minind = np.argmin(comp) if min(comp)<min(complh) else np.argmin(complh) #find lowest shift among right and left handed
	helixPoints2order = np.roll(np.argsort(comp),-list(np.argsort(comp)).index(minind)) #sort points to start at the lowest shift

	testfig = False
	if testfig:
		fig=plt.figure()
		ax=plt.axes(projection='3d')
		ax.set_box_aspect((1,1,1))
		for pt in helixPointsComp1:
			ax.scatter(pt[0],pt[1],pt[2],color='orange')
		[ax.text(p[0],p[1],p[2],ind) for ind,p in enumerate(helixPointsComp1)]
		for pt in helixPointsComp2:
			ax.scatter(pt[0],pt[1],pt[2],color='black')
		[ax.text(p[0],p[1],p[2],ind) for ind,p in enumerate(helixPointsComp2)]


	target=revolutions%1*2*np.pi#shift*2*np.pi/connectivity #target rotation minus whole revs
	targetIdx = np.argmin([abs(c-target) for c in comp]) #closest to the target rotation, right handed

	helixPoints2order=np.roll(helixPoints2order,shift)
	helixPointsComp2 = helixPointsComp2[helixPoints2order]


	compAngles = [signedAngleBetweenVectors(helixPointsComp1[n],val,point) for n,val in enumerate(helixPointsComp2)]
	compAngles = [c+2*np.pi if abs((c+2*np.pi)-target) < abs(c-target) else c for c in compAngles]
	maxShift=2*np.pi/(connectivity)   #i.e. to shift up to or more than an extra strand


	helixPoints1 = helixPoints1[helixPoints1order] #reorder to right-handed
	helixPoints2 = helixPoints2[helixPoints2order] #reorder to right-handed, matching to nearest helixPoint1


	bar.nodeEdges = [[node1.edgeListByPoint[idxAtNode1][n] for n in helixPoints1order],
					 [node2.edgeListByPoint[idxAtNode2][n] for n in helixPoints2order]]
		#edges w/r/t node for the current bar, in the corrected order


	helixAngles = wholerevs*2*np.pi*np.ones(connectivity)
	for n in range(np.shape(helixAngles)[0]):
		# print('pointcomp',helixPoints1[n]-point,helixPoints2[n]+point)
		helixAngles[n] = helixAngles[n] + compAngles[n]


	helixLength = (scipy.spatial.distance.euclidean(bar.end,bar.start)
		- np.sqrt(sphereRadius1**2 - helixRadius**2) - np.sqrt(sphereRadius2**2 - helixRadius**2))


	anglesOfStrands = np.zeros(connectivity) #that is, angles of strands w/r/t where they get created
	adjustedRevolutions = revolutions + (helixAngles %(2*np.pi))/(2*np.pi)

	pathLength=np.sqrt((2*np.pi*min(adjustedRevolutions)*helixRadius)**2+
					   helixLength**2)
	numberOfPoints=math.floor(pathLength/sampling)  #floor to make sampling >= for abq purposes
	rotateToPointMatrix = rotationMatrixVectorToVector(np.array([0,0,1]),point)
	rotateToPointInverse = scipy.linalg.inv(rotationMatrixVectorToVector(np.array([0,0,1]),point))
	orientedHelices = [[] for _ in range(connectivity)]
	pitchParameter = [[] for _ in range(connectivity)]

	for strand in range(connectivity):
		pitchParameter[strand] = helixLength / helixAngles[strand] #(2.0 * np.pi * adjustedRevolutions[strand])
		helixOrigin = np.array([0,0,1])*np.sqrt(sphereRadius1**2 - helixRadius**2)
		helix = makeBaseHelix(helixRadius,helixLength,helixAngles[strand],numberOfPoints,np.sqrt(sphereRadius1**2 - helixRadius**2))
		helixAtPoint= np.dot(rotateToPointMatrix,helix.T).T
		v1 = cartesianToSpherical(helix[0,:])
		v2 = cartesianToSpherical(np.dot(rotateToPointInverse,helixPoints1[strand,:].T).T)

		anglesOfStrands[strand] = v2[2]-v1[2] #rotate strand into place around the helix axis
		rot = scipy.spatial.transform.Rotation.from_rotvec(anglesOfStrands[strand]*point)
		orientedHelix = rot.apply(helixAtPoint)
		orientedHelices[strand] = orientedHelix[1:-1] + node1.centerPoint #originPoint
	return(orientedHelices,adjustedRevolutions,helixLength)





def wovenJoint(node, bars, revolutions,sampling): #helices

	"""
	III: For each node, use the helix and tangent paths to connect
	"""

	numPts = np.shape(node.unitPts)[0]
	pairs = node.pairs
	numPairs = np.shape(pairs)[0]
	helixAngles = [np.arctan2(node.strutLengths[pt], (2.0 * np.pi * revolutions[pt] * node.ptRadii[pt])) for pt in range(node.unitPts.shape[0])]

	neighborsListByPair = [[] for _ in range(numPairs)]
	helixAnglesByPair = [[] for _ in range(numPairs)]

	for n in range(numPts):
		neighborArrays = np.asarray(node.neighborTangentList[n][:]) #list of neighbors, in an arbitrary order
		radialAngles = np.pi/2 - np.asarray(node.neighborTangentAngleList[n][:]) #list of angles of tangent curves, in same order
		referenceCenters = np.asarray(node.neighborCenterList[n][:]) #list of centers, in same order
		referencePairs = np.asarray(node.edgeListByPoint[n][:]) #list of pairs that these helices belong to, in same order
		connectivity = np.shape(referencePairs)[0]
		edgeRevs = np.array(revolutions[n]) #list of revolutions, in same order

		if np.shape(edgeRevs)==():
			edgeRevs = edgeRevs*np.ones(np.shape(node.ptRadii))


		try:
			helixLen = bars[n].helixLength
		except:
			helixLen = bars[n].strutLength - 2*np.sqrt(node.sphereRadius**2-bars[n].barRadius**2)

		edgeAngles = np.arctan2(helixLen, (1 * np.pi * edgeRevs * node.ptRadii[n])) # (2 * np.pi * edgeRevs * node.ptRadii[n]))

		# for each neighboring point, find the connecting strand between the neighbor-path center point and the bottom of the helix
		for m in range(connectivity):
			currentPair = referencePairs[m]
			neighborsListByPair[currentPair].append(neighborArrays[m,:])
			helixAnglesByPair[currentPair].append(edgeAngles[m]) #helixAngles[m])


	numPairs = np.shape(pairs)[0]
	outputPath=[]
	for n in range(numPairs):
		edge = pairs[n,:]
		edgePoints = cartesianToSpherical(np.array(neighborsListByPair[n]))
		edgeAngles = np.array(helixAnglesByPair[n])
		edgeNeighbors = neighborsListByPair[n] #this
		tanPathLength = node.sphereRadius*angleBetweenVectors(edgeNeighbors[0][:],edgeNeighbors[1][:])
		curvePtNumber = math.floor(tanPathLength/sampling)+2 #+endpoints
		s = list(linspace(0,1,curvePtNumber))
		tangentPath = cartesianToSpherical(scipy.spatial.geometric_slerp(edgeNeighbors[0][:], edgeNeighbors[1][:], s))

		bp = scipy.interpolate.BPoly.from_derivatives([0, 1], [[edgePoints[0,0], -edgeAngles[0]], [edgePoints[1,0], edgeAngles[1]]]) #this
		interpolatedRadii = np.zeros((1,curvePtNumber))
		for i in range(curvePtNumber):
			interpolatedRadii[:,i] = bp(s[i])
		tangentPath[:,0] = interpolatedRadii
		outputPath.append(sphericalToCartesian(tangentPath)*node.sphereRadius+node.centerPoint)
	return(outputPath)





def shiftCell(point,vectors,shifts):
	"""
	Shift list of points by linear combination of vectors for tessellation
	"""
	transformedPoint = np.array(point)+ shifts[0]*vectors[0][0] + shifts[1]*vectors[1][0] + shifts[2]*vectors[2][0]
	return(list(transformedPoint))





def makeLattice(latticeTransformationVectors,originalLatticePoints,originalLatticeBars,latticeCells,showUC=False): #unrepeatedPoints,
	"""
	Generate original lattice points based on input dictionary values, lattice size
	"""
	allLatticePoints = list(originalLatticePoints.copy())#unrepeatedPoints.copy()


	#latticeCells
	originalLatticeBarPoints = [[originalLatticePoints[bar[0]],originalLatticePoints[bar[1]]] for bar in originalLatticeBars]
	allLatticeBarPoints = originalLatticeBarPoints.copy()
	inLattice = [1 for point in allLatticePoints]
	originsByPoint = [[(0,0,0),ind] for ind,point in enumerate(originalLatticePoints.copy())]

	runningTotal = len(allLatticePoints)
	originsInLat = list(product(range(latticeCells[0]),range(latticeCells[1]),range(latticeCells[2])))
	for origin in list(product(range(-1,latticeCells[0]+1),range(-1,latticeCells[1]+1),range(-1,latticeCells[2]+1))):
		if origin == (0,0,0):
			continue
		latticeBarsAtOrigin = [[shiftCell(bar[0],latticeTransformationVectors,origin),
			shiftCell(bar[1],latticeTransformationVectors,origin)] for bar in originalLatticeBarPoints]
		allLatticeBarPoints.extend(latticeBarsAtOrigin)
		unitCellAtOrigin = [shiftCell(point,latticeTransformationVectors,origin) for point in originalLatticePoints]
		allLatticePoints.extend(unitCellAtOrigin)
		isInLattice = int(origin in originsInLat)
		inLattice.extend([isInLattice for point in unitCellAtOrigin])
		originsByPoint.extend([[origin,pind] for pind,pt in enumerate(unitCellAtOrigin)])
		print(origin)
		runningTotal = len(allLatticePoints)


	#TODO: make allLatticeBars before allLatticeBarPoints so it's easier to sort only the unique bars
	# 	first. and/or, make allLatticeBars and allLatticePoints sets.

	allLatticePoints,idxs,inv = np.unique(allLatticePoints,axis=0,return_index=True,return_inverse=True) #okay
	inLatticeCompact=[[] for pt in allLatticePoints]
	originsCompact=[[] for pt in allLatticePoints]
	for id in range(len(allLatticePoints)):
		inLatticeCompact[id]=np.array(inLattice)[np.where(inv==id)[0]].any() #because otherwise points get sorted away
		originsCompact[id] = [originsByPoint[val] for val in np.where(inv==id)[0]]
	allLatticeBarPoints = np.unique(allLatticeBarPoints,axis=0)


	#fill in list of bars in index form:

	allLatticeBars=np.full((allLatticeBarPoints.shape[0:2]),-1,dtype='int')

	pointlen = len(allLatticeBarPoints)
	for n,pt in enumerate(allLatticeBarPoints): #new
		allLatticeBars[n,0] = np.nonzero([a.all() for a in allLatticePoints==pt[0,:]])[0][0]
		allLatticeBars[n,1] = np.nonzero([a.all() for a in allLatticePoints==pt[1,:]])[0][0]
		print(n,'/',pointlen)

	#sort out redundant bars in index form then remove from point form:
	allLatticeBars = np.delete(allLatticeBars,np.where(allLatticeBars==[-1,-1])[0], axis=0)
	allLatticeBars,idxs = np.unique(np.sort(allLatticeBars,axis=1),axis=0,return_index=True)
	allLatticeBarPoints = allLatticeBarPoints[idxs]

	if showUC:
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.set_box_aspect((1,1,1))
		axlength = 1
		for point in allLatticePoints:
			ax.scatter(point[0],point[1],point[2])
		for bar in allLatticeBarPoints:
			ax.plot(bar[:,0],bar[:,1],bar[:,2])
		plt.show()

	return(allLatticePoints,allLatticeBarPoints,allLatticeBars,inLatticeCompact,originsCompact)




def sortStrands(nodes,bars):
	"""
	Connect separated strands
	"""
	nodelist=set([node for node in nodes if nodes[node].isReal])
	barlist=set([bar for bar in bars if bars[bar].helix is not None])
	edgelist=set([edgepair for sublist in [[(node,idx) for idx,pair in enumerate(nodes[node].pairs)]
										   for node in nodelist] for edgepair in sublist])

	curves = []
	while edgelist: #while there remain edges (node, edge index) in the total list
		edge = edgelist.pop()
		node = edge[0] #the node is the first part of the 'edge'
		idx,pair = edge[1],nodes[node].pairs[edge[1]] #second part of the edge is the pair index on the node in Node.pairs
		startingNode,startingPair,startingIdx=node,pair,idx
		# startPath=nodes[node].path[:,:,idx]  #path along this edge
		startPath = nodes[node].path[idx][:,:]
		curve = Curve(startPath,[node])  #make a curve of that path
		for direction in [0,1]: #going both directions; 0 is backwards, 1 is forwards
			barPair = [nodes[startingNode].bars[p] for p in startingPair]
			node,pair,idx = startingNode,startingPair,startingIdx
			# print(direction)
			if nodes[node].bars[pair[direction]] in barlist: #if the bar going from the start of the pair exists
				thisBar = barPair[direction]
				while bars[thisBar].helix is not None: #while the current bar has a helix on it
					curve.bars.append(thisBar)
					thisNode= bars[thisBar].nodeIdxs.index(node) #idx of starting node@bar #i.e. does it flip or not
					thatNode= int(not(thisNode)) #the other idx, of course
					edgeLocOnBar = bars[thisBar].nodeEdges[thisNode].index(idx)  #idx of strand/edge on starting node
					curve.addPath(bars[thisBar].helix[edgeLocOnBar],thisNode == direction, not direction)
					nextNode=bars[thisBar].nodeIdxs[thatNode] #node at the far end of this bar
					nextEdge=bars[thisBar].nodeEdges[thatNode][edgeLocOnBar] #edge on said node connecting to this bar

					#now find the edge at the new node:
					if (nextNode,nextEdge) in edgelist:
						node = nextNode
						curve.nodes.append(node)
						edge = (nextNode,nextEdge)
						edgelist.remove(edge)
						idx,pair = edge[1],nodes[node].pairs[edge[1]]
						barPair = [nodes[node].bars[p] for p in pair]  #new pair of bars - the previous one, and a new one at the end

						thatBar = int(not barPair.index(thisBar))
						thisBar = barPair[thatBar] #this should be the new bar
						curve.addPath(nodes[node].path[idx][:,:],not thatBar==direction,not direction)
					else:
						break
				curves.append(curve)
	return curves


def saveBeamOrigins(curves,filename='beams_origin.csv'):
	import csv
	with open(filename,'w',newline='') as f:
		f.close()
	for index,curve in enumerate([curve for curve in curves if curve is not None]):
		if curve.isBeam:
			with open(filename,'a',newline='') as f:
				writer=csv.writer(f)
				writer.writerows([["%.4f"%p for p in point]+["%d"%c for c in curve.origin[0]] for point in curve.path])
	print('Beams saved to ', filename)
	return




def exportCSV(curves,latticeCells,L,packageBeams=True,doubleNetwork=False,trim=False,zeroout=False,folder=[]):
	"""
	Save folder of CSV files for each curve, and one file containing beams if double network
	"""
	import csv
	from datetime import date
	import os

	if not folder:
		folder = 'wovenCSV_' +  date.today().isoformat()
	beamfilename=folder+'/beams.csv'
	if not os.path.exists(folder):
		os.makedirs(folder)
	if doubleNetwork: #empty beam file
		with open(beamfilename,'w',newline='') as f:
			f.close()
	for index,curve in enumerate([curve for curve in curves if curve is not None]):
		filename= folder+'/'+"{:06d}".format(index)+'.csv' #"{:02d}".format(number)
		if curve.isBeam and packageBeams:
			if doubleNetwork:
				# [p1,p2] = curve.path
				filename = beamfilename
				with open(filename,'a',newline='') as f: #b') as f:
					writer = csv.writer(f)
					writer.writerows([["%.4f"%p for p in point] for point in curve.path])
				f.close()
					#if separate1D, move endpoint towards the other endpoint in that direction
					#if separate3D, move endpoint towards the other endpoint along endpt-startpt vector

		else:
			with open(filename,'w',newline='') as f: #b') as f:
				writer = csv.writer(f)
				trimpt = L#23.1
				mins = [-trimpt,-trimpt,-trimpt]
				maxes = [latticeCells[0]*trimpt*2-trimpt,latticeCells[1]*trimpt*2-trimpt,latticeCells[2]*trimpt*2-trimpt]
				if trim:
					if zeroout:
						writer.writerows([["%.4f" % (float(p)-mins[ind]) for ind,p in enumerate(point)] for point in curve.path if
										  mins[0]<=point[0]<=maxes[0]
										  and mins[1]<=point[1]<=maxes[1]
										  and mins[2]<=point[2]<=maxes[2]])
					else:
						writer.writerows([["%.4f"%float(p) for ind,p in enumerate(point)] for point in curve.path if
										  mins[0]<=point[0]<=maxes[0]
										  and mins[1]<=point[1]<=maxes[1]
										  and mins[2]<=point[2]<=maxes[2]])
				else:
					writer.writerows([["%.4f"%p for p in point] for point in curve.path])
			f.close()
	print('Files saved to', folder)
	return

def exportOneCSV(curves,latticeCells,L,filename=[]):
	"""
	Save single CSV file with all points
	"""
	import csv
	from datetime import date
	if not filename:
		filename='wovenCSV_'+date.today().isoformat()+'.csv'
	curvelengths=[len(curve.path) for curve in curves]
	ncurves=len(curvelengths)
	ptlist=[curve.path for curve in curves]
	mins=[-L,-L,-L]
	maxes=[latticeCells[0]*L*2-L,latticeCells[1]*L*2-L,latticeCells[2]*L*2-L]
	box=np.array(maxes)-np.array(mins)

	with open(filename,'w',newline='') as f:
		writer=csv.writer(f,delimiter=' ')
		writer.writerows([[str(ncurves)]])
		writer.writerows([["%.4f"%p for p in box]])
		writer.writerows([["%i"%p for p in curvelengths]])
		for curve in curves:
			writer.writerows([["%.4f"%p for p in point] for point in curve.path])
		# writer.writerows([[["%.4f"%p for p in point] for point in curve.path] for curve in curves])
	f.close()
	print('File saved to',filename)
	return

def plotCurves(curves,latticeCells):
	"""
	Plot lattice
	"""
	fig=plt.figure()
	ax=plt.axes(projection='3d')
	ax.set_box_aspect(latticeCells)
	for ind,curve in enumerate(curves):
		if curve is not None:
			path = curve.path #[curve.path[:,2]>=L]
			ax.plot(path[:,0],path[:,1],path[:,2])#,alpha=.2)#,color='black')
	ax.set_xlabel('x')
	plt.show()
	return


def plotLatticePoints(latticeCells,latticeDict,latName):
	ax = plt.axes(projection='3d')
	ax.set_box_aspect(latticeCells)#(1,1,1))
	[ax.scatter(p[0],p[1],p[2]) for p in latticeDict[latName].baseLatticePoints]
	[ax.text(p[0],p[1],p[2],ind) for ind,p in enumerate(latticeDict[latName].baseLatticePoints)]
	return


def save_piped_stl(curves,rad,pts,lengthtol=[],filename='woven_mesh.stl',eccentricity=1):

	"""
	Save stl
	Keyword arguments:
		 lengthtol -- tolerance for removing short segments (default radius/2)
		 filename -- .stl filename (default 'woven_mesh.stl')
	"""
	from matplotlib import cm,tri
	from stl import mesh

	if not lengthtol:
		lengthtol = rad/2
	lp = list(np.linspace(0,2*np.pi,endpoint=True,num=pts+1))
	sines = np.sin(lp)
	coses = np.cos(lp)
	u0 = list(range(pts+1))

	meshes = []
	for curve in curves:
		curve = np.array(curve.path)
		curve=[[c[0],c[1],c[2]/eccentricity] for c in curve]
		d = np.diff(curve,n=1,axis=0)
		n =  np.linalg.norm(d,axis=1)
		n = np.append(n,[lengthtol+1]) #keep end point regardless
		n[0] = lengthtol+1 #keep start point regardless
		curve2 = [c for ind,c in enumerate(curve) if n[ind]>lengthtol] #remove short lengths
		surf = np.zeros((len(curve2)+2,pts+1,3)) #len, circpts, 3
		surf[0] = np.tile(curve2[:][0],(pts+1,1)) #circpts, 3
		surf[-1] = np.tile(curve2[:][-1],(pts+1,1)) #
		surf[-2] = np.tile(curve2[:][-1],(pts+1,1)) #

		dv = np.diff(curve2,n=1,axis=0)
		ninp = [1,0,0]
		if np.dot(dv[0],ninp)>.95: ninp = [0,1,0]
		nv2 = ninp
		for it in range(len(curve2)-1):
			nv = normalize(np.cross(nv2,dv[it]))
			nv2 = normalize(np.cross(dv[it],nv))
			offsets = np.array([rad*coses[c]*nv + rad*sines[c]*nv2 for c in range(pts+1)])
			surf[it+1,:,:] = np.array([curve[it]+offset for offset in offsets])
		surf[len(curve2),:,:] = [curve[-1]+offset for offset in offsets]
		u,v = np.meshgrid(u0,list(range(len(curve2)+2)))
		u,v = u.flatten(), v.flatten()
		tris = tri.Triangulation(u,v)
		x,y,z = np.ravel(surf[:,:,0]),np.ravel(surf[:,:,1]),np.ravel(surf[:,:,2])
		data = np.zeros(len(tris.triangles), dtype=mesh.Mesh.dtype)
		curve_mesh = mesh.Mesh(data, remove_empty_areas=False)
		curve_mesh.x[:] = x[tris.triangles]
		curve_mesh.y[:] = y[tris.triangles]
		curve_mesh.z[:] = z[tris.triangles]*eccentricity
		meshes.append(curve_mesh)
		# ax.plot_trisurf(x,y,z,triangles = tris.triangles)
	output_mesh = mesh.Mesh(np.concatenate([m.data for m in meshes]))
	output_mesh.save(filename)
	print('STL saved to',filename)
	return(output_mesh)

###################################################
	#Plotly figure

#
# def plotly_plot_curves(curves,L):
# 	fig=go.Figure()
# 	for index, curve in enumerate([curve for curve in curves if curve is not None]):
# 		path = np.array(curve.path)
# 		x, y, z = path[:,0], path[:,1], path[:,2]
#
# 		fig.add_trace(go.Scatter3d(
# 			x=x+L, y=y+L, z=z+L,
# 			mode='lines',
# 			name=f'Curve {index}'
# 		))
#
# 	fig.update_layout(
# 		width=1000,
# 		height=600,
# 		scene=dict(
# 			xaxis=dict(
# 				backgroundcolor="#262626",
# 				color="#626262",  # Set the axis and label color
# 				gridcolor="#626262",  # Set the grid color if grid is visible
# 				showgrid=True,
# 				zeroline=True
# 			),
# 			yaxis=dict(
# 				backgroundcolor="#262626",
# 				color="#626262",  # Set the axis and label color
# 				gridcolor="#626262",  # Set the grid color if grid is visible
# 				showgrid=True,
# 				zeroline=True
# 			),
# 			zaxis=dict(
# 				backgroundcolor="#262626",
# 				color="#626262",  # Set the axis and label color
# 				gridcolor="#626262",  # Set the grid color if grid is visible
# 				showgrid=True,
# 				zeroline=True
# 			),
# 			xaxis_title='X Axis',
# 			yaxis_title='Y Axis',
# 			zaxis_title='Z Axis'
# 		),
# 		plot_bgcolor="#262626",
# 		paper_bgcolor="#262626",
# 		legend=dict(
# 			font=dict(
# 				color="#626262"  # Set legend text color
# 			)
# 		)
# 	)
# 	html_filename = "3d_plot.html"
# 	fig.write_html(html_filename)
# 	return(fig,html_filename)
# 		##visibleDeprecationWarning
#
#

####################################################################################################################
# Classes
####################################################################################################################
# 


class Lattice:
	#defines lattice base format for latticeDict
	def __init__(self,name,latticeTransformationVectors,baseLatticePoints,baseLatticeBars):
		self.name = name
		self.latticeTransformationVectors= latticeTransformationVectors
		self.baseLatticePoints = baseLatticePoints
		self.baseLatticeBars = baseLatticeBars
		self.scaledVectors = latticeTransformationVectors
		self.scaledPoints = baseLatticePoints
		self.dual = None
	def __repr__(self):
		cl = self.__class__.__name__
		return f'{cl}({self.name},{self.latticeTransformationVectors},{self.baseLatticePoints},{self.baseLatticeBars})'
	def make(self,latticeCells,L):
		self.scaledVectors = [[L*v[0],v[1]] for v in self.latticeTransformationVectors]
		self.scaledPoints = L*np.array(self.baseLatticePoints)
		latticePoints,latticeBarPoints,latticeBars,inLattice,originsByPoint=makeLattice(self.scaledVectors,
																						self.scaledPoints,
																						self.baseLatticeBars,
																						latticeCells)
		return(latticePoints,latticeBarPoints,latticeBars,inLattice,originsByPoint)


class Node:
	"""
	Node
	"""

	def __init__(self,isReal,ind,unitPts,centerPoint,bars,sphereRadius,ptRadii,strutLengths,
		revolutions=None,helixPtNumber=None,curvePtNumber=None,pointIdxs=None,points=None):
		self.isReal = isReal
		self.idx = ind
		self.unitPts = unitPts #unit points around the node
		self.initialSphereRadius = sphereRadius
		self.ptRadii = ptRadii
		self.strutLengths = strutLengths
		self.revolutions = revolutions
		self.centerPoint = centerPoint
		self.bars = bars #bars around the node corresponding to unit points
		self.pointIdxs = pointIdxs
		self.point = points
		self.edgeListByPoint = None
		if isReal:
			(sphereRadiusNew,neighborTangentList,neighborTangentAngleList,neighborCenterList,edgeListByPoint,barListByPoint,neighborListByPoint,
				pairs,strutLengths) = wovenUnitCell(unitPts,bars,sphereRadius,ptRadii,strutLengths,revolutions,helixPtNumber,curvePtNumber)
			self.ptRadii = ptRadii
			self.strutLengths = strutLengths
			self.sphereRadius = sphereRadiusNew
			self.neighborTangentList = neighborTangentList
			self.neighborTangentAngleList = neighborTangentAngleList
			self.neighborCenterList = neighborCenterList
			self.edgeListByPoint = edgeListByPoint
			self.barListByPoint = barListByPoint
			self.neighborListByPoint = neighborListByPoint
			self.anglesPerBar = list(zip(neighborTangentAngleList,bars))
			self.pairs = pairs
	#TODO: function for finding the bar and other end?

class Bar:
	"""
	Bar (edge)
	"""
	#TODO: make sure all attributes are stated explicitly in class Bar
	def __init__(self,ind,nodeIdxs,position,strutParams,L,helix=None):
		self.idx = ind
		self.nodeIdxs = nodeIdxs
		self.position = position
		self.barRadius, self.barRevolutions = strutParams(position,L)
		if self.barRadius<0:
			raise ValueError('Negative Radius.')
		self.helix=helix
		self.pitchParameter=None



class Curve:
	"""
	Curve containing path points
	"""
	def __init__(self,path=[],nodes=[],nodePoints=[],bars=[],barStrands=[],origin=[],isBeam=False):
		self.barStrands = barStrands
		self.bars = bars
		self.nodePoints = nodePoints
		self.nodes = nodes
		self.path = path
		#TODO: make sure all attributes are stated in class Curve
		self.isBeam=isBeam
		self.origin=origin
		self.curvature = None
		self.pathLength = None
	def addPath(self,addition,flip,addToStart=0):
		if flip:
			addition = np.flip(addition,axis=0)
		if addToStart:
			self.path=np.concatenate((addition,self.path),axis=0)
		else:
			self.path = np.concatenate((self.path,addition),axis=0) #self.path.extend(newPath)
	def closePath(self,tol):
		if np.allclose(self.path[0],self.path[-1],atol=tol):
			self.path=np.concatenate((self.path,[self.path[0]]),axis=0)

	def computeCurvature(self):
		x_t = np.gradient(self.path[0,:])
		y_t = np.gradient(self.path[1,:])
		z_t = np.gradient(self.path[2,:])
		x_tt = np.gradient(x_t)
		y_tt = np.gradient(y_t)
		z_tt = np.gradient(z_t)
		pathLen =np.shape(self.path)[0]
		kappa = np.zeros((pathLen,1))
		for t0 in range(pathLen):
			gradNorm = np.linalg.norm([x_t[t0],y_t[t0],z_t[t0]])
			kappa[t0] = np.linalg.norm(np.cross([x_t[t0],y_t[t0],z_t[t0]],[x_tt[t0],y_tt[t0],z_tt[t0]]))/gradNorm**3
		self.curvature=kappa

	def computePathLength(self):
		self.pathLength = np.sum([scipy.spatial.distance.euclidean(p,self.path[ind-1]) for ind,p in enumerate(self.path)])

####################################################################################################################
# Lattices
####################################################################################################################


def wovenLattice(latName,L,strutParams,sampling,latticeCells=[1,1,1],doubleNetwork=False,defectList=[],defectPoints=[],exportCSV=0,exportOneCSV=0,trim=True):
	"""
	Main
	"""



	latticeDict = {}
	cubicVecs = [[np.array([2.,0.,0.]),[0]],[np.array([0.,2.,0.]),[0]],[np.array([0.,0.,2.]),[0]]]

	latticeDict['bcc'] = Lattice('bcc',cubicVecs,
								 [[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1],[0,0,0]],
								 [[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8]])


	latticeDict['octahedron'] = Lattice('octahedron',cubicVecs,
										[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,-1],[0,0,1]],
										[[0,1],[1,2],[2,3],[3,0],[0,5],[1,5],[2,5],[3,5],[0,4],[1,4],[2,4],[3,4]])
	latticeDict['cubic'] = Lattice('cubic',cubicVecs,
								   [[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1]],
								   [[0,3],[0,1],[1,5],[3,5],[3,4],[5,7],[1,6],[0,2],[4,7],[7,6],[6,2],[2,4]])
	latticeDict['tetrakai'] = 	Lattice('tetrakai',cubicVecs,
										 [[.5,0,1],[0,.5,1],[0,-.5,1],[-.5,0,1],[.5,0,-1],[0,.5,-1],[0,-.5,-1],[-.5,0,-1],
										  [1,.5,0],[1,0,.5,],[1, 0,-.5],[1,-.5,0],[-1,.5,0],[-1,0,.5],[-1, 0,-.5],[-1,-.5,0],
										  [.5,1,0],[0,1,.5],[0,1,-.5,],[-.5,1,0],[.5,-1,0],[0,-1,.5,],[0,-1,-.5],[-.5,-1,0]],
										 [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],
										   [20,21],[21,23],[23,22],[22,20],[12,13],[13,15],[15,14],[14,12],
										   [16,17],[17,19],[19,18],[18,16],[8,9],[9,11],[11,10],[10,8],
										   [0,9],[1,17],[2,21],[3,13],[4,10],[5,18],[6,22],[7,14],
										   [8,16],[12,19],[15,23],[20,11]])
	latticeDict['diamond'] = Lattice('diamond',	cubicVecs,
									 [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1],
									  [-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1],
									  [-.5,-.5,-.5],[.5,.5,-.5], [-.5,.5,.5],[.5,-.5,.5]],
									 [[2,16],[5,17],[0,14],[7,15],[13,16],[13,17],[11,17],[11,15],[12,16],
									  [12,15],[10,14],[10,15],[8,14],[8,16],[9,14],[9,17]])
	latticeDict['cuBCC'] = Lattice('cuBCC',cubicVecs,
								   [[-.5,.5,.5],[.5,-.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[-.5,.5,.5],[.5,-.5,.5],[.5,.5,-.5],[.5,.5,.5],
									[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1]],
								   [[0,3],[0,1],[1,5],[3,5],[3,4],[5,7],[1,6],[0,2],[4,7],[7,6],[6,2],[2,4],
									[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]])

	co=np.cos(np.pi/3)
	si=np.sin(np.pi/3)
	latticeDict['tetracubic'] = Lattice('tetracubic',[[np.array([2,0.,0.]),[0]],[np.array([-2*co,2*si,0.]),[0]],[np.array([0.,0.,2.]),[0]]],
										np.array([[0,0,0],[2,0,0],[2+2*co,2*si,0],[2*co,2*si,0],[0,0,2],[2,0,2],[2+2*co,2*si,2],[2*co,2*si,2]]),
										[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],
										 [7,4],[0,4],[1,5],[2,6],[3,7],[1,3],[5,7]])


	latticeDict['octahedron'].dual = 'bcc'
	latticeDict['bcc'].dual = 'octahedron'

	aa=1.3
	latticeDict['arrow']=Lattice('arrow',cubicVecs,
								 [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[0,0,-1+aa],[0,0,1+aa]],
								 [[0,4],[1,4],[2,4],[3,4],[0,5],[1,5],[2,5],[3,5]])


	latticeDict['octet']=Lattice('octet',cubicVecs,
								 [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,-1],[0,0,1], #0 1 2 3 4 5
								  [-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,-1],[1,1,1]], # 6-13
								 [[0,1],[1,2],[2,3],[3,0],[0,5],[1,5],[2,5],[3,5],[0,4],[1,4],[2,4],[3,4],
								  [0,7],[0,11],[0,12],[0,13],
								  [1,8],[1,10],[1,12],[1,13],
								  [2,6],[2,8],[2,9],[2,10],
								  [3,6],[3,7],[3,9],[3,11],
								  [4,6],[4,7],[4,8],[4,12],
								  [5,9],[5,10],[5,11],[5,13],])

	np.seterr(all='raise')




	dualBeams = False #future: for automating the dual-type double network
	packageBeams = True #put all beams in a file called beams.csv, instead of listing them as 2-point curves
	# trim =  True #clip to unit cell; bad for ucs with beams along edge (cubic, tetrakai)
	# defectList=[]
	# defectList = [(0,0,0),(2,0,0),(4,0,0),(6,0,0),(8,0,0),(10,0,0),(12,0,0),(14,0,0),(16,0,0),
	# 			  (1,0,1),(3,0,1),(5,0,1),(7,0,1),(9,0,1),(11,0,1),(13,0,1),(15,0,1),(17,0,1),
	# 			  (0,0,2),(2,0,2),(4,0,2),(6,0,2),(8,0,2),(10,0,2),(12,0,2),(14,0,2),(16,0,2),
	# 			  (1,0,3),(3,0,3),(5,0,3),(7,0,3),(9,0,3),(11,0,3),(13,0,3),(15,0,3),(17,0,3),
	# 			  (0,0,4),(2,0,4),(4,0,4),(6,0,4),(8,0,4),(10,0,4),(12,0,4),(14,0,4),(16,0,4),
	# 			  (1,1,0),(3,1,0),(5,1,0),(7,1,0),(9,1,0),(11,1,0),(13,1,0),(15,1,0),(17,1,0),
	# 			  (0,1,1),(2,1,1),(4,1,1), (6, 1, 1),(8, 1, 1),(10, 1, 1),(12, 1, 1), (14, 1, 1),(16,1,1),
	# 			  (1,1,2),(3,1,2),(5,1,2),(7, 1, 2),(9, 1, 2),(11, 1, 2),(13, 1, 2),(15, 1, 2),
	# 			  (0,1,3),(2,1,3),(4,1,3),(6, 1, 3),(8, 1, 3),(10, 1, 3),(12, 1, 3),(14, 1, 3),
	# 			  (1,1,4),(3,1,4),(5,1,4),(7, 1, 4),(9, 1, 4),(11, 1, 4),(13, 1, 4),(15, 1, 4),
	# 			  (0,2,0),(2,2,0),(4,2,0),(6, 2, 0),(8, 2, 0),(10, 2, 0),(12, 2, 0),(14, 2, 0),(16, 2, 0),
	# 			  (1,2,1),(3,2,1),(5,2,1),(7, 2, 1),(9, 2, 1),(11, 2, 1),(13, 2, 1),(15, 2, 1),
	# 			  (0,2,2),(2,2,2),(4,2,2),(6, 2, 2),(8, 2, 2),(10, 2, 2),(12, 2, 2),(14, 2, 2),(16, 2, 2),
	# 			  (1,2,3),(3,2,3),(5,2,3),(7, 2, 3),(9, 2, 3),(11, 2, 3),(13, 2, 3),(15, 2, 3),
	# 			  (0,2,4),(2,2,4),(4,2,4),(6, 2, 4),(8, 2, 4),(10, 2, 4),(12, 2, 4),(14, 2, 4),(16, 2, 4),
	# 			  (1,3,0),(3,3,0),(5,3,0),(7, 3, 0),(9, 3, 0),(11, 3, 0),(13, 3, 0),(15, 3, 0),
	# 			  (0,3,1),(2,3,1),(4,3,1),(6, 3, 1),(8, 3, 1),(10, 3, 1),(12, 3, 1),(14, 3, 1),(16, 3, 1),
	# 			  (1,3,2),(3,3,2),(5,3,2),(7, 3, 2),(9, 3, 2),(11, 3, 2),(13, 3, 2),(15, 3, 2),
	# 			  (0,3,3),(2,3,3),(4,3,3),(6, 3, 3),(8, 3, 3),(10, 3, 3),(12, 3, 3),(14, 3, 3),(16, 3, 3),
	# 			  (1,3,4),(3,3,4),(5,3,4),(7, 3, 4),(9, 3, 4),(11, 3, 4),(13, 3, 4),(15, 3, 4),
	# 			  ]


	# defectPoints = []#[0,2]#[8] #[0,2] for octa # [8] for bcc
	# defectPoints = []
	gap=2.
	zGap=True #need true for bcc
	inputSphereRadius = L/10
	latticePoints,latticeBarPoints,latticeBars,inLattice,originsByPoint = latticeDict[latName].make(latticeCells,L)
	latticeConnectionsByPoint, latticeBarsByPoint = sortNeighbors(len(latticePoints), latticeBars) #(bar indices by node), point positions of node
	maxConnections = np.amax([len(pointconns) for pointconns in latticeConnectionsByPoint]) #'real' aka connections when not missing any
	nodes = {}
	bars = {}
	numLatticePts = latticePoints.shape[0]

	notch=False
	if notch:
		removeCells ={(0,0,2),(0,1,2),(0,2,2),(1,0,2),(1,1,2),(1,2,2)}#,(2,0,2),(2,1,2),(2,2,2)}
		borderCells = {(0,0,3),(0,0,1),(0,1,3),(0,1,1),(0,2,3),(0,2,1),
					   (1,0,3),(1,0,1),(1,1,3),(1,1,1),(1,2,3),(1,2,1)}
		for pt,org in enumerate(originsByPoint):
			inLattice[pt] = False if (set([a for [a,b] in org]).intersection(removeCells)
									  and not set([a for [a,b] in org]).intersection(borderCells)) else inLattice[pt]

	showUC = False
	if showUC:
		ax = plt.axes(projection='3d')
		ax.set_box_aspect(latticeCells)#(1,1,1))
		[ax.scatter(p[0],p[1],p[2]) for p in latticeDict[latName].baseLatticePoints]
		[ax.text(p[0],p[1],p[2],ind) for ind,p in enumerate(latticeDict[latName].baseLatticePoints)]
		plt.show()

	####################################################################################################################
	# Define Node Parameters
	####################################################################################################################

	sphereRadius = float(inputSphereRadius)
	# helices = []
	helixPtNumber = 60 #points in helix
	curvePtNumber = 15 #points in curve

	####################################################################################################################
	# Set up nodes and find tangent points
	####################################################################################################################

	barAngles = [[] for _ in range(len(latticeBars))]
	neighborsForAngles = [[] for _ in range(len(latticeBars))]
	barNeighborIdxs = [[] for _ in range(len(latticeBars))]

	#going through all the points
	for latticePtIdx in range(numLatticePts):
		thisPointCenterPoint = latticePoints[latticePtIdx] #thisPoint
		thisPointBars = latticeBarsByPoint[latticePtIdx] #bars connected to thisPoint
		thisPointPointIdxs = latticeConnectionsByPoint[latticePtIdx] #nodes connected to thisPoint
		thisPointPoints = np.array([latticePoints[conn]-thisPointCenterPoint for conn in thisPointPointIdxs]) #relative positions connected to thisPoint
		strutPositions = [[(latticePoints[latticePtIdx,0]+latticePoints[conn][0])/2,
			(latticePoints[latticePtIdx,1]+latticePoints[conn][1])/2,
			(latticePoints[latticePtIdx,2]+latticePoints[conn][2])/2] for conn in latticeConnectionsByPoint[latticePtIdx]]
		strutLengths = [scipy.linalg.norm(pt) for pt in thisPointPoints]
		ptRadii = []
		revolutions = []

		#now produce bars, if they don't already exist, for each bar connected to the thisPoint
		for n in range(len(thisPointBars)):            #jbar in range(np.shape(thisPointBars)[0]):
			if thisPointBars[n] not in bars:
				bars[thisPointBars[n]] = Bar(thisPointBars[n],[latticePtIdx,thisPointPointIdxs[n]],strutPositions[n],strutParams,L)
				bars[thisPointBars[n]].strutLength = scipy.spatial.distance.euclidean(latticePoints[latticePtIdx],
																					  latticePoints[thisPointPointIdxs[n]])
				bars[thisPointBars[n]].strands = np.shape(strutLengths)[0] #TODO: NO!
			ptRadii.append(bars[thisPointBars[n]].barRadius)
			revolutions.append(bars[thisPointBars[n]].barRevolutions)


		unitPts = np.array([point/scipy.linalg.norm(point) for point in thisPointPoints])

		#now produce nodes for each node that has full connections
		if inLattice[latticePtIdx]: #and len(latticeConnectionsByPoint[latticePtIdx]) == maxConnections:
		# if len(latticeConnectionsByPoint[latticePtIdx]) == maxConnections:
			nodes[latticePtIdx] = Node(True,latticePtIdx,unitPts,thisPointCenterPoint,thisPointBars,sphereRadius,ptRadii,
				strutLengths,revolutions,helixPtNumber,curvePtNumber,thisPointPointIdxs,thisPointPoints)
			for n in range(len(thisPointBars)):
				barNeighborIdxs[thisPointBars[n]].append(nodes[latticePtIdx].barListByPoint[n])
				barAngles[thisPointBars[n]].append(list(nodes[latticePtIdx].neighborTangentAngleList[n]))
				neighborsForAngles[thisPointBars[n]].append(nodes[latticePtIdx].neighborTangentList[n])


		else:
			nodes[latticePtIdx] = Node(False,latticePtIdx,unitPts,thisPointCenterPoint,thisPointBars,sphereRadius,ptRadii,strutLengths)

	count = 0
	barAngleDiff = []




	####################################################################################################################
	# Make helices
	####################################################################################################################



	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.set_box_aspect(latticeCells)#(1,1,1))
	# axlength = 2


	for barIdx in range(len(latticeBars)):
		if not barAngles[barIdx]:
			barAngles[barIdx] = [[0,0,0],[0,0,0]] #[[0,0,0,0],[0,0,0,0]]
			continue
		if np.shape(barAngles[barIdx])[0] == 2:
			try:
				[node1,node2] = [nodes[val] for val in latticeBars[barIdx]]
			except:
				print('oops')
				continue
			[bars[barIdx].startNode,bars[barIdx].endNode] = latticeBars[barIdx] #endpoints in index
			[bars[barIdx].start,bars[barIdx].end] = latticeBarPoints[barIdx] #endpoints in space
			bars[barIdx].center = (bars[barIdx].start + bars[barIdx].end)/2 #center in space
			[bars[barIdx].neighbors1,bars[barIdx].neighbors2] = neighborsForAngles[barIdx] #neighbor tangent points for each neighboring node, in space
			[bars[barIdx].neighborAngles1,bars[barIdx].neighborAngles2]	= barAngles[barIdx] #neighbor tangent angles for each neighboring node
			[bars[barIdx].neighborNodes1,bars[barIdx].neighborNodes2]	= barNeighborIdxs[barIdx]
			strands = np.shape(bars[barIdx].neighbors1)[0]
			bars[barIdx].angleDiff = np.array(
				np.roll(barAngles[barIdx][1],int(bars[barIdx].barRevolutions*strands)))+np.array(
				barAngles[barIdx][0])

			bars[barIdx].barStrutLength = scipy.spatial.distance.euclidean(bars[barIdx].start,bars[barIdx].end) - (node1.sphereRadius+node2.sphereRadius)

			#TODO: test path length formulation
			# pathLength = np.sqrt((2*np.pi*bars[barIdx].barRevolutions*bars[barIdx].barRadius)**2 +
			# 					 bars[barIdx].barStrutLength**2)
			# helixPtNumber = floor(pathLength/sampling) #floor to make sampling >= for abq purposes
			startPoint = node1.unitPts[node1.bars.index(barIdx)]

			if (node1.isReal and node2.isReal and len(node1.neighborTangentList[node1.bars.index(barIdx)])>0
					and len(node2.neighborTangentList[node2.bars.index(barIdx)])>0):
				helix1,adjustedRevolutions,helixLength= makeRotatedHelices(bars[barIdx],startPoint,sampling,node1,node2)
				bars[barIdx].barRevolutions = adjustedRevolutions
				bars[barIdx].helixLength = helixLength
				helix1 = np.array(helix1)
				bars[barIdx].helix = np.array(helix1)
				count+=1

	print(count, 'complete bars with helices')




	####################################################################################################################
	# Make joints
	####################################################################################################################

	count = 0
	count2 = 0
	count3 = 0

	for nodeIdx in nodes:
		node = nodes[nodeIdx]
		# print(node.bars)
		try:
			nodeBars = [bars[barIdx] for barIdx in node.bars]
			revolutions = [bars[barIdx].barRevolutions for barIdx in node.bars]
		except:
			count3 +=1
			continue

		if node.edgeListByPoint:
			revs = [bars[bar].barRevolutions for bar in node.bars]
			node.path = wovenJoint(node, nodeBars, revs,sampling)
			count+=1
		else:
			count2+=1

	print(count, 'complete nodes, ', count2, 'incomplete nodes, ', count3, 'incomplete bar nodes, ', count+count2+count3, 'total nodes?')














	curves = sortStrands(nodes,bars)
	beams = []
	if doubleNetwork:
		for ind,beam in enumerate(latticeBars):
			if inLattice[beam[0]] and inLattice[beam[1]]: #inLattice: is the point at this index in the lattice
				p1,p2 = latticePoints[beam[0]].copy(),latticePoints[beam[1]].copy()
				intArra=[a[0] for a in originsByPoint[beam[0]]]
				intArrb=[a[0] for a in originsByPoint[beam[1]]]
				defInta = set([a[0] for a in originsByPoint[beam[0]]]).intersection(set(defectList))
				defIntb = set([a[0] for a in originsByPoint[beam[1]]]).intersection(set(defectList))
				origin=list(set(intArra).intersection(set(intArrb)))
				if defInta and defInta.intersection(defIntb):
					intList = list(defInta.intersection(defIntb))[0]
					defIndexa = list([ind for ind,a in enumerate(intArra) if a==intList])[0] #which index is the point in the defective cell
					defIndexb = list([ind for ind,a in enumerate(intArrb) if a==intList])[0]
					if originsByPoint[beam[0]][defIndexa][1] in defectPoints:
						if zGap:
							ab= np.sign(latticePoints[beam[1]][2]-latticePoints[beam[0]][2])
							moveVec = np.array([0,0,ab*gap])
						else:
							ab=np.array(latticePoints[beam[1]])-np.array(latticePoints[beam[0]])
							dist=np.linalg.norm(ab)
							moveVec=ab*(gap/dist)#(1-(dist-gap)/dist)
						p1 += moveVec
					if originsByPoint[beam[1]][defIndexb][1] in defectPoints:
						if zGap:
							ab=np.sign(latticePoints[beam[0]][2]-latticePoints[beam[1]][2])
							moveVec=np.array([0,0,ab*gap])
						else:
							ab=np.array(latticePoints[beam[0]])-np.array(latticePoints[beam[1]])
							dist=np.linalg.norm(ab)
							moveVec=ab*(gap/dist)#ab*(1-(dist-gap)/dist)
						p2 += moveVec
				pts = np.array([p1, p2])
				curves.append(Curve(pts, beam, pts, ind, origin=origin, isBeam=True)) #path,nodes,nodePoints,bars

	tol = .01
	for ind,curve in enumerate(curves): #close closed paths
		curve.closePath(tol)

	for ind,curve in enumerate(curves): #delete repeat curves
		for ind2,curve2 in enumerate(curves):
			if not ind==ind2 and curve.path.any==curve2.path.any:
				del(curves[ind2])

	return(curves)





