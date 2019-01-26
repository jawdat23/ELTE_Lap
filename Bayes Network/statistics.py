from naive_bayes import NaiveBayes



folder = 'E:\\elte\\1\\lab\\'
bank_file_name = 'bank-data.csv'
cars_file_name = 'cars.data'
data_length = 600
cars_data_length = 1728
data_length = cars_data_length


nb = NaiveBayes()
nb.training_percentage = 1

nb_generated = NaiveBayes()
nb_generated.training_percentage = 1

nb_generated_dep = NaiveBayes()
nb_generated_dep.training_percentage = 1



nb.read_data(folder + bank_file_name)
nb.remove_columns(['id'])
nb.split_contentious_data(['age','income'], 'pep',3)

nb_generated.read_data(folder + 'generated-bank-data.csv')
nb_generated_dep.read_data(folder + 'generated-bank-data-depedencies.csv')


for attribute in nb.df.columns:
    print('default',attribute,nb.df[attribute].value_counts()/data_length)
    print('generated',attribute, nb_generated.df[attribute].value_counts() / data_length)
    print('generated_dep',attribute, nb_generated_dep.df[attribute].value_counts() / data_length)

print(nb.df['income'])
print(nb.df['age'])
import matplotlib.pyplot as plt
plt.plot(nb.df['income'])
plt.plot(nb.df['age'])
plt.ylabel('some numbers')
plt.show()

plt.plot(nb_generated.df['income'])
plt.plot(nb_generated.df['age'])
plt.ylabel('some numbers')
plt.show()

plt.plot(nb_generated_dep.df['income'])
plt.plot(nb_generated_dep.df['age'])
plt.ylabel('some numbers')
plt.show()

