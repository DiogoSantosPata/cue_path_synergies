import numpy as np
from hippocampus import Hippocampus
import socket
import os


UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP


def sender( index, msg):
	message_string = 0
	if index != '5.0':  message_string = np.around( (msg - msg.min())/(msg.max() - msg.min()), decimals=2)
	else: message_string = [ msg[0]*4 , msg[1]*4 ] 
	message_string = np.hstack( ( [index], message_string) )
	message_string = ";".join( message_string.astype(str)  )
	sock.sendto(message_string, (UDP_IP, UDP_PORT))


def main():
	DGNumCell=400

	xx = np.loadtxt('trajectory/simulatedX.txt')
	yy = np.loadtxt('trajectory/simulatedY.txt')

	overlap_condition = 0
	
	overlap_dir_name = '../log_data/overlap_'+str(overlap_condition) 
	os.mkdir( overlap_dir_name )

	for density_condition in range(10,-1,-1):
		print "Density: ", density_condition

		overlap_density_dir_name = '../log_data/overlap_'+str(overlap_condition)+'/density_'+str(density_condition)
		os.mkdir( overlap_density_dir_name )


		for phase in range(3): ## 0) learn/save weights;  1) load weights, no noise;  2) load weights, with noise
			print "Phase: ", phase

			hippocampus = Hippocampus(overlap_condition, density_condition, phase, DGNumCell=400)

			log_mec, log_dg = [],[]

			for ii in range(1,1000):#for ii in range(1,len(yy)):

				####################
				## Representation ##
				####################
				speedVector = np.complex(xx[ii]-xx[ii-1] , yy[ii]-yy[ii-1] )
				hippocampus.update_hippocampus(speedVector, [xx[ii],yy[ii]] )

				###########
				## Learn ##
				###########
				if phase == 0:
					if np.where( hippocampus.get_DG_activity() > 29. ) > DGNumCell-5  and  np.random.randint(0,100)==0:
						print 'HEBB'
						hippocampus.learn()

				#########
				## Log ##
				#########
				if phase != 0:
					log_dg.append(   hippocampus.DGactivity  )
					log_mec.append(  np.ravel(hippocampus.MEC_activity[:,:,0])  )

				#########
				## UDP ##
				#########
				msg_1 = np.hstack(  (np.ravel(hippocampus.MEC_activity[:,:,0]), 
									 np.ravel(hippocampus.MEC_activity[:,:,2]) ) )				
				sender( '1.0', msg_1)
				sender( '6.0', hippocampus.LEC_activity )
				sender( '2.0', hippocampus.DGactivity )
				sender( '5.0', [xx[ii] , yy[ii] ]  )

			##########
			## SAVE ##
			##########
			if phase == 0:
				np.save(overlap_density_dir_name+'/MEC_DG_Weights.npy', hippocampus.MEC_DG_Weights )
				np.save(overlap_density_dir_name+'/LEC_DG_Weights.npy', hippocampus.LEC_DG_Weights )
				np.save(overlap_density_dir_name+'/DG_MEC_Weights.npy', hippocampus.DG_MEC_Weights )
			if phase == 1:
				np.savetxt(overlap_density_dir_name+'/log_mec_baseline.txt', log_mec)
				np.savetxt(overlap_density_dir_name+'/log_dg_baseline.txt', log_dg)
			if phase == 2:
				np.savetxt(overlap_density_dir_name+'/log_mec.txt', log_mec)
				np.savetxt(overlap_density_dir_name+'/log_dg.txt', log_dg)

if __name__ == '__main__':
	main()