import numpy as np

class Hippocampus():

	def __init__(self, overlap_condition, density_condition, phase_number, DGNumCell=400):
		dir_name = '../log_data/overlap_'+str(overlap_condition)+'/density_'+str(density_condition)

		#MEC
		self.set_MEC_activity()

		self.grid_cell_noise = False
		if phase_number == 2: self.grid_cell_noise = True

		#LEC
		self.numb_of_lec_cells = 400
		self.possible_LEC_activity = np.load('../pre_sets_data/LEC_overlap_'+str(overlap_condition)+'.npy')[density_condition]
		self.possible_LEC_activity *= 30. 

		#DG
		self.DGNumCell = DGNumCell
		self.DGactivity = np.zeros(self.DGNumCell)

		#Weights
		if phase_number == 0 :
			self.MEC_DG_Weights = np.random.uniform(0,1,(self.mECLayers*self.mm*self.nn,self.DGNumCell))
			self.MEC_DG_Weights = self.normalizeWeight(self.MEC_DG_Weights)

			self.LEC_DG_Weights = np.random.uniform(0,1,(self.numb_of_lec_cells,self.DGNumCell))
			self.LEC_DG_Weights = self.normalizeWeight(self.LEC_DG_Weights)

			self.DG_MEC_Weights = np.random.uniform(0,1,(self.DGNumCell,self.mECLayers*self.mm*self.nn))
			self.DG_MEC_Weights = self.normalizeWeight(self.DG_MEC_Weights)

		else:
			self.MEC_DG_Weights = np.load(dir_name+'/MEC_DG_Weights.npy' )
			self.LEC_DG_Weights = np.load(dir_name+'/LEC_DG_Weights.npy' )
			self.DG_MEC_Weights = np.load(dir_name+'/DG_MEC_Weights.npy' )
			


		#E%max
		self.emax = 0.95


	def update_hippocampus(self, speedVector, position=None):
		"""
			Update hippocampal populations.
			speedVector input is complex number.
		"""

		## Update grid cells MEC II ##
		mecActTemp = []
		for jj in range(0,self.mECLayers):
			rrr = self.mECGain[jj]*np.exp(1j*0)
			matWeights = self.update_MEC_Weights(self.distTri,speedVector,rrr)
			activityVect = np.ravel(self.MEC_activity[:,:,jj])
			activityVect = self.Bfunc(activityVect, matWeights)
			activityTemp = activityVect.reshape(self.mm,self.nn)
			activityTemp += self.TAO *( activityTemp/np.mean(activityTemp) - activityTemp)
			activityTemp[activityTemp<0.] = 0.
			self.MEC_activity[:,:,jj] = (activityTemp-np.min(activityTemp))/(np.max(activityTemp)-np.min(activityTemp)) * 30.

			#NOISE
			if self.grid_cell_noise == True:
				self.MEC_activity[:,:,jj] += np.random.uniform(-10.,10., self.MEC_activity[:,:,jj].shape )


		## Update LEC cells ##
		self.LEC_activity = np.copy( self.possible_LEC_activity[ position[0]/10 , position[1]/10 ] )

		## Update place cells Dentate Gyrus ##
		if np.max( self.LEC_activity ) > 0.0:
			self.DGactivity = .5*self.Cfunc(self.MEC_activity,self.MEC_DG_Weights) + .5*self.Cfunc(self.LEC_activity,self.LEC_DG_Weights)
		else:
			self.DGactivity = self.Cfunc(self.MEC_activity,self.MEC_DG_Weights)
		
		self.DGactivity = self.Emax(self.DGactivity)

		## Update MEC cells again with DG feedback ##
		# self.MEC_activity = self.Cfunc( self.DGactivity, self.DG_MEC_Weights )



	def set_MEC_activity(self):
		self.TAO = 0.9
		self.II = 0.3
		self.SIGMA = 0.24
		self.SIGMA2 = self.SIGMA**2
		self.TT = 0.05
		self.mm,self.nn = 20,20

		self.mECGain =  [0.02, 0.01, 0.03, 0.06, 0.09, 0.1]
		self.mECLayers = len(self.mECGain)	
		
		self.MEC_activity = np.load('../pre_sets_data/initial_mec.npy') #np.random.uniform(0,1,(self.mm,self.nn,self.mECLayers))

		self.distTri = self.buildTopology(self.mm,self.nn)
		self.vvvVo = 0
		self.checkDir = []


	def buildTopology(self,mm,nn):
		mmm = (np.arange(mm)+(0.5/mm))/mm
		nnn = ((np.arange(nn)+(0.5/nn))/nn)*np.sqrt(3)/2
		xx,yy = np.meshgrid(mmm, nnn)
		posv = xx+1j * yy
		Sdist = [ 0+1j*0, -0.5+1j*np.sqrt(3)/2, -0.5+1j*(-np.sqrt(3)/2), 0.5+1j*np.sqrt(3)/2, 0.5+1j*(-np.sqrt(3)/2), -1+1j*0, 1+1j*0 ]
		
		xx,yy = np.meshgrid( np.ravel(posv) , np.ravel(posv) )
		distmat = xx-yy

		for ii in range(len(Sdist)):
			aaa1 = abs(distmat)
			rrr = xx-yy + Sdist[ii]
			aaa2 = abs(rrr)
			iii = np.where(aaa2<aaa1)
			distmat[iii] = rrr[iii]
		return distmat.transpose()

	def update_MEC_Weights(self,topology,speedVector,rrr):
		matWeights = self.II * np.exp((-abs(topology-rrr*speedVector)**2)/self.SIGMA2) - self.TT
		return matWeights

	def Bfunc(self,activity, matWeights):
		""" A general dot sum update function. """
		activity += np.dot(activity,matWeights)
		return activity

	def normalizeWeight(self, matWeights):
		""" Normalizing the weights from EC to DG. Don't use this method in case you apply De Almeida 2009a weight distribution. """
		matWeights /= np.tile(np.linalg.norm(matWeights,axis=0),[matWeights.shape[0],1])
		return matWeights

	def learn_MEC_DG(self):
		"""	Hippocampus learnMECToDG: this method updates the synaptic strength between grid and DG place cells. """
		lrate = 0.7#0.1#0.7
		rrra = np.ravel(self.DGactivity)/np.max(self.DGactivity)
		rrrb = np.ravel(self.MEC_activity)/np.max(self.MEC_activity)
		ccaa,ccmm = np.meshgrid(rrra,rrrb)
		rrrc = ccaa*ccmm
		self.MEC_DG_Weights += lrate*(rrrc-self.MEC_DG_Weights)
		self.MEC_DG_Weights = self.normalizeWeight(self.MEC_DG_Weights)

	def learn_LEC_DG(self):
		"""	Hippocampus learnMECToDG: this method updates the synaptic strength between grid and DG place cells. """
		if np.max(self.LEC_activity) > 0.0:
			lrate = 0.7#0.1#0.7
			rrra = np.ravel(self.DGactivity)/np.max(self.DGactivity)
			rrrb = np.ravel(self.LEC_activity)/np.max(self.LEC_activity)
			ccaa,ccmm = np.meshgrid(rrra,rrrb)
			rrrc = ccaa*ccmm
			self.LEC_DG_Weights += lrate*(rrrc-self.LEC_DG_Weights)
			self.LEC_DG_Weights = self.normalizeWeight(self.LEC_DG_Weights)


	def learn_DG_MEC(self):
		"""	Hippocampus learn_DG_MEC: this method updates the synaptic strength between DG place cells and MEC grid cells. """
		lrate = 0.7
		rrra = np.ravel(self.MEC_activity)/np.max(self.MEC_activity)
		rrrb = np.ravel(self.DGactivity)/np.max(self.DGactivity)
		ccaa,ccmm = np.meshgrid(rrra,rrrb)
		rrrc = ccaa*ccmm
		self.DG_MEC_Weights += lrate*(rrrc-self.DG_MEC_Weights)
		self.DG_MEC_Weights = self.normalizeWeight(self.DG_MEC_Weights)



	def learn(self):
		self.learn_MEC_DG()
		self.learn_LEC_DG()
		# self.learn_DG_MEC()


	def Cfunc(self,Activity,Weights):
		mmma = np.ravel(Activity)
		Activity = np.dot(mmma,Weights)
		# mmma = np.max(Activity)
		# Activity -= self.emax*mmma
		# Activity[Activity<0] = 0
		return Activity

	def Emax(self, Activity):
		mmma = np.max(Activity)
		Activity -= self.emax*mmma
		Activity[Activity<0] = 0
		return Activity




	def get_MEC_activity(self):
		return np.ravel(self.MEC_activity)

	def get_DG_activity(self):
		return self.DGactivity