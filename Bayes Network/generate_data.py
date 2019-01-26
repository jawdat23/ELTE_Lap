from naive_bayes import NaiveBayes



folder = 'E:\\elte\\1\\lab\\'
bank_file_name = 'bank-data.csv'
cars_file_name = 'cars.data'

cars_data_length = 1728

nb = NaiveBayes()
nb.training_percentage = 1



nb.read_data(folder + bank_file_name)
try:
    nb.remove_columns(['id'])
except:
    pass
nb.split_contentious_data(['age','income'], 'pep',3)

nb.build_model('pep',[])
# nb.build_model('quality',[])

# for attribute in nb.df.columns:
#     print(attribute,nb.df[attribute].value_counts()/600)
#     if attribute == 'age':
#         continue
# res = nb.df.groupby(['pep', "age"]).size()


inctances_number = 600
output_file = folder + 'generated-bank-data.csv'
# inctances_number = cars_data_length
# output_file = folder + 'generated-cars.csv'
generated_data = nb.generate_data(inctances_number, output_file)

output_file = folder + 'generated-bank-data-depedencies.csv'
dependencies = {}
# dependencies['lug_boot'] = ['doors']
dependencies['income'] = ['age']


generated_data = nb.generate_data_for_correlated_attributes(inctances_number, output_file, dependencies)

