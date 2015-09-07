"""
	LEC cells activity for set of density/overlap conditions
	Arena: 100 x 100 
	Bins:  10 x 10
	Generate 10 maps of increasing densities at overlap condition 0
"""
import numpy as np

number_of_lec_cells = 400
lec_ref = np.random.uniform(0,1,(10,10, number_of_lec_cells)) # referencia for lec maxima density

densities = []

densities.append( lec_ref  ) # Full density 

for condition in range(1,10):
	tmp_lec = np.copy(lec_ref)
	tmp_lec[ np.random.randint(0,10,condition*10) , np.random.randint(0,10,condition*10) ] *= 0.0 #np.zeros(number_of_lec_cells)
	densities.append( tmp_lec  )

densities.append( np.copy(lec_ref)*0.0  ) # NO density 

np.save('LEC_overlap_0', densities)