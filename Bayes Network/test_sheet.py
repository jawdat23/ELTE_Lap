from naive_bayes import NaiveBayes

bank_file_name = r'E:\elte\1\lab\bank-data.csv'
cars_file_name = r'E:\elte\1\lab\generated-bank-data-depedencies.csv'

error = 0
for k in range(0,10):
    nb = NaiveBayes()
    nb.training_percentage = 0.8

    nb.read_data(cars_file_name)
    # nb.remove_columns(['id'])
    # nb.split_contentious_data(['age','income'], 'pep',3)

    # nb.build_model('pep',[ 'age','income'])
    nb.build_model('pep',[])
    # nb.build_model('quality',[])

    results = nb.classify()
    correct=0
    for i in range(0,len(results)):
        if results[i] == nb.testing_labels[i]:
            correct+=1
    print('iteration ', k+1, ': ',correct/len(results))
    error += correct/len(results)

print('last: ',error/(k+1) )