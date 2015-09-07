import numpy as np
import matplotlib.pyplot as plt

## Arena size
arenaX = [0,100]
arenaY = [0,100]
## Initial position
xlist = [arenaX[1]/2]
ylist = [arenaY[1]/2]

def conv(ang):
	x = np.cos(np.radians(ang)) #+ np.random.uniform(-1,1)
	y = np.sin(np.radians(ang)) #+ np.random.uniform(-1,1)
	return x , y

def random_navigation(length):
	thetaList = []

	theta = 90
	counter = 0
	lenght_counter = 0
	for i in range(length):
	#while lenght_counter < length and (xlist[0]+5<xlist[-1]<xlist[0]+5  and  ylist[0]+5<ylist[-1]<ylist[0]+5 ):
		lenght_counter += 1

		prevTheta = theta
		## X
		if( xlist[-1]>2 ):
			theta=theta
		else:
			theta = np.random.randint(-85,85)

		if( xlist[-1]<98 ):
			theta=theta
		else:
			theta = np.random.randint(95,260)
		## Y
		if( ylist[-1]>2 ):
			theta=theta
		else:
			theta = np.random.randint(10,170)
		if( ylist[-1]<98 ):
			theta=theta
		else:
			theta = np.random.randint(190,350)


		# xlist.append( xlist[-1]+conv(theta)[0] + np.random.uniform(-0.5,0.5) )
		# ylist.append( ylist[-1]+conv(theta)[1] + np.random.uniform(-0.5,0.5)  )
		xlist.append( xlist[-1]+conv(theta)[0] )
		ylist.append( ylist[-1]+conv(theta)[1] )


		cx = abs( xlist[-1] - xlist[-2]  )
		cy = abs( ylist[-1] - ylist[-2]  )
		h = np.sqrt( cx**2 + cy**2  )
		counter+=h

		if(theta != prevTheta or i == length-1):
			thetaList.append( [prevTheta, conv(prevTheta)[0], conv(prevTheta)[1], counter]  )
			counter = 0


		
	np.savetxt('simulatedX.txt', xlist)
	np.savetxt('simulatedY.txt', ylist)

	plt.plot(xlist,ylist, '-')
	plt.show()

def linear_navigation(length):
	
	direction = 1
	random_y = 0
	for i in range(1,length):
		if xlist[i-1] >= arenaX[1]:
			direction = -1
		if xlist[i-1] <= 0:
			direction = 1
		xlist.append(  xlist[i-1]+direction )
		random_y += np.random.uniform(-0.5,0.5)
		ylist.append(  ylist[0]+random_y   )
		
	np.savetxt('simulated_linear_X.txt', xlist)
	np.savetxt('simulated_linear_Y.txt', ylist)

	plt.plot(xlist,ylist, '-')
	plt.ylim(0,100)
	plt.show()

def main():
	random_navigation(10000)
	#linear_navigation(2000)

if __name__ == '__main__':
	main()