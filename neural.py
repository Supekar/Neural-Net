import numpy as np
import csv
import numpy as np
import math
import sys
import copy
ytrain=[]
ytest=[]

train_file=sys.argv[1]
test_file=sys.argv[2]
train_label=sys.argv[3]
test_label=sys.argv[4]
metrics_out=sys.argv[5]
epochs=int(sys.argv[6])
hidden_unit=int(sys.argv[7])
init_flag=int(sys.argv[8])
learning_rate=float(sys.argv[9])

train_entropy=[]
test_entropy=[]

with open(train_file ,mode='r') as input:
	inputdata = list(csv.reader(input))

	direct=[]
	for x in inputdata:
		direct.append(x)


#print(direct)
a = np.asarray(direct, dtype = float)
#print(a)
inputs=a[:,1:]
#print(inputs)
label=a[:,:1]
label = np.asarray(label, dtype = int)
inputs = np.asarray(inputs, dtype = int)
onehotlabel=np.zeros((inputs.shape[0],10))
for i in range(inputs.shape[0]):
	onehotlabel[i,label[i]]=1
#print(onehotlabel)
onehotlabel = np.asarray(onehotlabel, dtype = float)
inputs = np.asarray(inputs, dtype = float)
label = np.asarray(label, dtype = float)
#print(inputs.shape)
#print(label.shape)
hidden_units=hidden_unit
inputs1=np.insert(inputs, 0, float(1), axis=1)
#print(inputs1.shape)


with open(test_file ,mode='r') as input1:
	inputdatatest = list(csv.reader(input1))

	direct1=[]
	for x in inputdatatest:
		direct1.append(x)


#print(direct)
atest = np.asarray(direct1, dtype = float)
#print(a)
inputstest=atest[:,1:]
#print(inputs)
labeltest=atest[:,:1]
labeltest = np.asarray(labeltest, dtype = int)
inputstest = np.asarray(inputstest, dtype = int)
onehotlabeltest=np.zeros((inputstest.shape[0],10))
for i in range(inputstest.shape[0]):
	onehotlabeltest[i,labeltest[i]]=1
onehotlabeltest = np.asarray(onehotlabeltest, dtype = float)
#print(onehotlabel)
inputstest = np.asarray(inputstest, dtype = float)
labeltest = np.asarray(labeltest, dtype = float)
#print(inputs.shape)
#print(label.shape)
#hidden_units=4
inputs1test=np.insert(inputstest, 0, float(1), axis=1)



epoch=epochs
flag=init_flag
learn_rate=learning_rate

if flag==2:
	alpha=np.zeros((hidden_units,inputs.shape[1]), dtype=int)
	alpha1=np.insert(alpha, 0, float(0), axis=1)
	beta=np.zeros((10,hidden_units), dtype=int)
	beta1=np.insert(beta, 0, float(0), axis=1)
	#print(beta1.shape)

	#print(alpha1)
	#print(alpha1.shape)
	#print(beta1)
	#print(beta1.shape)

elif flag==1:
	alpha = np.random.uniform(low=-0.1, high=0.1, size=(hidden_units,inputs.shape[1]))
	alpha1=np.insert(alpha, 0, float(0), axis=1)
	#print()
	beta = np.random.uniform(low=-0.1, high=0.1, size=(10,hidden_units))
	beta1=np.insert(beta, 0, float(0), axis=1)
	#print(beta1.shape)
	#print(alpha1)
	#print(alpha1.shape)
	#print(beta1)
	#print(beta1.shape)
count=0
counttest=0
count1=0.0
counttest1=0.0

for i in range(epoch):
	traintotal=0.0
	testtotal=0.0
	
	for j in range(inputs1.shape[0]):
		#print(inputs1
		
		a=np.dot(alpha1, inputs1[j].T)
		z=1.0/(1+np.exp(-a.T))
		#print(z)
		z1=np.insert(z, 0, 1.0)
		b=np.dot(beta1,z1)
		ycap=np.exp(b.T)/np.sum(np.exp(b.T))
		
		
		
		
		y=np.argmax(ycap)
		
		ylabel=np.zeros((1,10), dtype=float)
		#print(y)
		ylabel[0][y]=1
		#if list(ylabel[0])!=list(onehotlabel[j]):
			#count+=1.0

		x1=np.array([ycap-onehotlabel[j]])
		z1=np.array([z1])
		
		dbydb=np.dot(x1.T,z1)

		#print(dbydb.shape)
		#print(dbydb)
		


		beta1temp=np.delete(beta1,0,axis=1)
		inter1=np.dot(x1,beta1temp)
		z2=np.delete(z1,0,axis=1)
		
		inter2=z2*(float(1)-z2)
		inter3=inter1*inter2
		dbyda=np.dot(inter3.T,np.array([inputs1[j]]))
		alpha1=alpha1-learn_rate*dbyda
		beta1=beta1-learn_rate*dbydb
	#print(alpha1)
	#print(beta1)

	for j in range(inputs1.shape[0]):
		#print(inputs1
		
		a=np.dot(alpha1, inputs1[j].T)
		z=1.0/(1+np.exp(-a.T))
		#print(z)
		z1=np.insert(z, 0, float(1))
		b=np.dot(beta1,z1)
		ycap=np.exp(b.T)/np.sum(np.exp(b.T))
		y=np.argmax(ycap)
		#if int(i)==int(epoch):

			#ytrain.append(y)
		#print(y)
		#print(y)
		ylabel=np.zeros((1,10), dtype=float)
		#print(y)
		ylabel[0][y]=1
		#if j==1:
			#print(inputs1[0].shape)
			#print(alpha1.shape)
			#print(a.shape)
			#print(z.shape)
			#print(z1.shape)
			#print(beta1.shape)
			#print(b.shape)
			#print(ycap.shape)
			#print(b)
			#print(ycap)
			#print(y)


		if float(y)!=float(label[j]):
			#print(float(y))
			#print(float(label[j]))
			count+=1
		if float(y)==float(label[j]):
			count1+=1
		
		traintotal-=np.sum(np.multiply(np.log(ycap),onehotlabel[j]))
		
	#print(alpha1)
	#print(beta1)
	


	for j in range(inputs1test.shape[0]):
		#print(inputs1
		
		a=np.dot(alpha1, inputs1test[j].T)
		z=1.0/(1+np.exp(-a.T))
		#print(z)
		z1=np.insert(z, 0, float(1))
		b=np.dot(beta1,z1)
		ycap=np.exp(b.T)/np.sum(np.exp(b.T))
		y=np.argmax(ycap)
		#if int(i)==int(epoch):
			#ytest.append(y)
		#print(y)
		ylabel=np.zeros((1,10), dtype=float)
		#print(y)
		ylabel[0][y]=1
		if float(y)!=float(labeltest[j]):
			counttest+=1.0
		if float(y)==float(labeltest[j]):
			counttest1+=1

		
		
		testtotal-=np.sum(np.multiply(np.log(ycap),onehotlabeltest[j]))
	#print("epoch ")
	k=int(i+1)
	#print("trainentropy")
	err1=float(traintotal)/float(inputs1.shape[0])
	err1=round(err1,11)	
	train_entropy.append(err1)
	#print("epoch")
	#print(i+1)
	#print("testentropy")
	err2=float(testtotal)/float(((inputs1test.shape[0])))
	err2=round(err2,11)
	test_entropy.append(err2)

	#output.close()

train_err=float(count)/float(inputs1.shape[0]*epoch)

test_err=(float(counttest)/float(inputs1test.shape[0]*epoch))

for j in range(inputs1.shape[0]):
		#print(inputs1
		
		a=np.dot(alpha1, inputs1[j].T)
		z=1.0/(1+np.exp(-a.T))
		#print(z)
		z1=np.insert(z, 0, float(1))
		b=np.dot(beta1,z1)
		ycap=np.exp(b.T)/np.sum(np.exp(b.T))
		y=np.argmax(ycap)
		#if int(i)==int(epoch):

		ytrain.append(y)

for j in range(inputs1test.shape[0]):
		#print(inputs1
		
		a=np.dot(alpha1, inputs1test[j].T)
		z=1.0/(1+np.exp(-a.T))
		#print(z)
		z1=np.insert(z, 0, float(1))
		b=np.dot(beta1,z1)
		ycap=np.exp(b.T)/np.sum(np.exp(b.T))
		y=np.argmax(ycap)
		#if int(i)==int(epoch):
		ytest.append(y)

		

		
#print(ycap[0])
#print(np.log(ylabel[0]))
#print(onehotlabeltest[1])
#print(dbyda.shape)
#print(1-z2)

#print(beta1.shape)
#beta1temp=np.delete(beta1,0,axis=1)
#print(beta1temp)
#print(x1.shape)
#print(beta1temp.shape)

#print(z2.shape)
#print(inputs1.shape)
#print(beta1.shape)
#print(dbydb.shape)
#print(dbyda.shape)
#print(alpha1.shape)
#print(beta1.shape)
#print(inter1.shape)
#print(inter2.shape)
#print(inter3.shape)
#print(alpha1)
#print(beta1)

#print(count)
#print(counttest)
#print(count1)
#print(counttest1)
#print(alpha1)
#print(beta1)


#print(label)
#print(alpha1)		
#print(len(ytrain))	
#print(len(ytest))


with open(train_label, mode='w') as output1:

	#tsv_writer=csv.writer(output, delimiter='\t')
	for i in ytrain:
		output1.write(str(i))
		output1.write('\n')
		

with open(test_label, mode='w') as output2:

	#tsv_writer=csv.writer(output, delimiter='\t')
	for i in ytest:
		output2.write(str(i))
		output2.write('\n')	

with open(metrics_out, mode='w') as output:
	for i in range(len(train_entropy)):
		l=i+1

		
		output.write("epoch=" + str(l))
		
		output.write(" crossentropy(train): " + str(train_entropy[i]))
		output.write("\n")
		output.write("epoch=" + str(l))
		
		output.write(" crossentropy(test): " + str(test_entropy[i]))
		output.write("\n")

	output.write("error(train): " + str(train_err) )
		
		
	output.write("\n")
	output.write("error(test): " + str(test_err))
	

#with open(metrics_out, mode='w') as output:
		#output.write("error(train): " + str(train_err) )
		
		
		#output.write("\n")
		#output.write("error(test): " + str(test_err))
		
	
		
#output.close()

		

#print(b.shape)



		#print(a.shape)





