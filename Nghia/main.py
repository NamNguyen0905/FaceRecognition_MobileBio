''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import performance
import get_images
import get_landmarks

import pandas

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split

''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def classifier_process(clf):
    #split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    clf.fit(X_train, y_train)

    matching_scores = clf.predict_proba(X_test)

    gen_scores = []
    imp_scores = []
    classes = clf.classes_

    matching_scores = pandas.DataFrame(matching_scores, columns = classes)
    #test = pd.DataFrame()

    for i in range (len(y_test)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
    
    return gen_scores, imp_scores


# NB, SVM, ANN
image_directory = 'Project 1 Database'
#open the file and read the data in binary mode
# img_data = pickle.loads(open('image_data', "rb").read())
# print('Reading data...')
# X = img_data["images"]
# y = img_data["labels"]

X, y = get_images.get_images(image_directory)


''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y= get_landmarks.get_landmarks(X, y, "landmarks/", 68, False)



# create an instance of the classifier
clf = ORC(SVC(probability=True))
method = 'SVC'
gen_scores, imp_scores = classifier_process(clf)

clf2 = ORC(KNeighborsClassifier(3))
method2 = 'KNN(k = 3)'
gen_scores2, imp_scores2 = classifier_process(clf2)

clf3 = ORC(KNeighborsClassifier(7))
method3 = 'KNN(k = 7)'
gen_scores3, imp_scores3 = classifier_process(clf3)


#use classifiers for performance
performance.performance(gen_scores, imp_scores, gen_scores2, imp_scores2, gen_scores3, imp_scores3, 
                        method, method2, method3, 500)






'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
clf.fit(X_train, y_train)

matching_scores = clf.predict_proba(X_test)

gen_scores = []
imp_scores = []
classes = clf.classes_

matching_scores = pandas.DataFrame(matching_scores, columns = classes)
#test = pd.DataFrame()

for i in range (len(y_test)):
    scores = matching_scores.loc[i]
    mask = scores.index.isin([y_test[i]])
    
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])


clf2 = ORC(KNeighborsClassifier(5))
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.25, random_state = 42)
clf2.fit(X_train2, y_train2)

matching_scores2 = clf2.predict_proba(X_test2)

gen_scores2 = []
imp_scores2 = []
classes2 = clf2.classes_

matching_scores2 = pandas.DataFrame(matching_scores2, columns = classes2)
#test = pd.DataFrame()

for i in range (len(y_test2)):
    scores2 = matching_scores2.loc[i]
    mask2 = scores2.index.isin([y_test2[i]])
    
    gen_scores2.extend(scores2[mask2])
    imp_scores2.extend(scores2[~mask2])
    
performance.performance(gen_scores, imp_scores, gen_scores2, imp_scores2, 'SVC', 'Knn', 500)


'''


'''

num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []


for i in range(0, len(y)):
    query_img = X[i, :]
    query_label = y[i]
    
    #print('i = ', i)
    #print(query_img)

    template_imgs = np.delete(X, i, 0)
    template_labels = np.delete(y, i)
    
    '''
    # array = query_img
    # print(array.shape)
    # print(type(array))
    # array = np.reshape(array, (5, 5))
    # print(array.shape)
    # data = im.fromarray(array)
    
    # if data.mode != 'RGB':
    #     data = data.convert('RGB')
    #     data.save('./test_phtos/photo%d.jpg' %i)
'''
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_labels))
    y_hat[template_labels == query_label] = 1 
    y_hat[template_labels != query_label] = 0

    clf.fit(template_imgs, y_hat) # Train the classifier
    y_pred = clf.predict(query_img.reshape(1,-1)) # Predict the label of the query

    
    # Gather results
    if y_pred == 1:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)
        #print(labels_incorrect)
        #print(i)
        #plt.imshow(query_img.reshape(-1,1))
        #plt.show()

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))   
'''    