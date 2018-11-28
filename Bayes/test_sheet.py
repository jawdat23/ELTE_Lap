from naive_bayes import NaiveBayes

bank_file_name = r'G:\joodeee\elte\data science\1st\lab\task 2\bank-data.csv'
cars_file_name = r'G:\joodeee\elte\data science\1st\lab\task 2\cars.csv'

nb = NaiveBayes()
nb.training_percentage = 0.8

nb.read_data(cars_file_name)
#nb.remove_columns(['id', 'age','income'])

#nb.build_model('pep',[])
nb.build_model('quality',[])
results = nb.classify()

correct = 0
for i in range(0,len(results)):
    if results[i] == nb.testing_labels[i]:
        correct+=1
print(correct/len(results))