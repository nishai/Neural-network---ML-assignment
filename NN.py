from Network import *
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#	function to decode network output as the class prediction
def output_to_prediction(out):
	pred=[]
	for x in out:
		y = 1 if (x[0] > x[1]) else 0
		pred.append(y)
	return pred

def get_labels(test):
	y=[]
	for i in range(len(test)):
		y.append(net.feedforward(test_x[i]))
	return output_to_prediction(y)

def min_max_scale(list):
	min_x = min(list)
	max_x = max(list)
	for x in list:
		x = (x - min_x)/(max_x - min_x) - 0.5
	return list

"""READING AND STRUCTURING DATA"""
df = pd.read_csv('dataset.csv')

df.drop(['URL', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis=1, inplace=True)		#drop columns
df['CHARSET'] = df['CHARSET'].str.upper()
df['SERVER'] = df['SERVER'].str.upper()
df['WHOIS_COUNTRY'] = df['WHOIS_COUNTRY'].str.upper()

#convert columns with categorical data to enumerations
df['CHARSET'] = df['CHARSET'].astype('category')
df['SERVER'] = df['SERVER'].astype('category')
df['WHOIS_COUNTRY'] = df['WHOIS_COUNTRY'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df['CONTENT_LENGTH'].fillna(0, inplace=True)	# replace NaNs with 0s
df['DNS_QUERY_TIMES'].fillna(0, inplace=True)	# replace NaNs with 0s
# print(df.isnull().any())

#split into training, validation and test sets
train = df.sample(frac=0.6)
the_rest = df.loc[~df.index.isin(train.index), :] 
validation = the_rest.sample(frac=0.5) # 50% x 40% = 20%
test = train.loc[~the_rest.index.isin(validation.index), :] 

test_x, test_y = get_x_y(test)

scaler = MinMaxScaler()
test_x = scaler.fit_transform(test_x)
test_x = [x - 0.5 for x in test_x]

net = Network([16, 5, 2], train)
# init_weights = tuple(net.weights)
# init_biases = tuple(net.biases)
err0 = net.sgd()
# net.batch_size_as_percentage = 0.1
# net.weights = list(init_weights)
# net.biases = list(init_biases)
# err1 = net.sgd()
# net.batch_size_as_percentage = 0.4
# net.weights = list(init_weights)
# net.biases = list(init_biases)
# err2 = net.sgd()
# net.batch_size_as_percentage = 0.6
# net.weights = list(init_weights)
# net.biases = list(init_biases)
# err3 = net.sgd()

actual = get_labels(test)
predicted = output_to_prediction(test_y)
print(confusion_matrix(actual, predicted))
print(accuracy_score(actual, predicted))

# plt.plot(err0, label="Learning rate = 0.2")
# plt.plot(err1, label="Learning rate = 0.01")
# plt.plot(err2, label="Learning rate = 0.001")
# plt.plot(err3, label="Learning rate = 0.0001")
# plt.ylabel('Training error')
# plt.xlabel('Number of epochs')
# plt.show()