
# # Mini Project on Data-Mining

# # Title : Product Purchase Prediction using Social Network Ads

# # Presented By:

# ## Bagal Dipti
# ## Deokar Harshada


import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

df = pd.read_csv('Social_Network_Ads.csv')
df.head()
df.info()

df['Purchased'].value_counts()
df.drop('User ID' , axis = 1 , inplace=True)
df.head()






# Encoding of gender
# Get Dummies over Gender

gender = pd.get_dummies(df["Gender"] ,drop_first=True)

df = pd.concat([df ,gender] ,axis = 1)
df.head()
df.drop("Gender" , axis=1 , inplace=True)
df.head()


import seaborn as sns
sns.heatmap(df.corr(), cmap='coolwarm')
df.drop("Male", inplace=True, axis=1)
df.head()

# Transformation by Normalization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaled_array = ss.fit_transform(df.drop('Purchased' , axis = 1))

x = pd.DataFrame(data = scaled_array , columns=df.columns[:-1])
x.head()

y = df['Purchased']

# Spliting of dataset

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x ,y ,test_size = 0.3)



#  Prediction using Support Vector Machine (SVM)
def svm():
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    # Confusion Matrix

    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc = classification_report(y_test, y_pred)
    print(acc)
    acc1=accuracy_score(y_test,y_pred)*100
    print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
    #str = metrics.accuracy_score(y_test, y_pred) * 100
#    entry.delete(0, tk.END)
    # entry.insert(0, str)
    # Accuracy : 88.33%

    window=Tk()
    window.geometry('300x300')
    window.title("SVM")
    frm=Frame(window,width=300,height=100)
    frm.pack(side=TOP)
    lbx = Label(frm,text="Accuracy of Support Vector Machine",font=("Arial Bold",10)).grid(row=0,column=0)
    lby = Label(frm,text=acc1,font=("Arial Bold",10)).grid(row=0,column=1)
    window.mainloop()

    import scikitplot as skplot
    skplot.metrics.plot_confusion_matrix(y_test, y_pred)
    p = metrics.classification_report(y_test, y_pred, output_dict=True)
    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.show()

    #import seaborn as sns
   # sns.heatmap(cm, annot=True, annot_kws={"size": 15})  # font size

def con4():
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    skplt.metrics.plot_confusion_matrix(y_test ,y_pred)
    plt.show()

    # precision recall classification graph


    from yellowbrick.classifier import ClassificationReport, ClassificationScoreVisualizer

    # Instantiate the classification model and visualizer
    svm = SVC()
    classes = ['Purchased', 'Not Purchased']
    visualizer = ClassificationReport(svm, classes=classes, support=True)

    visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()

def PRS():

    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Precision recall curve Visualization

    from sklearn.metrics import precision_recall_curve, average_precision_score

    from inspect import signature


    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve for SVM: AP={0:0.2f}'.format(average_precision))

def HP():
    # hyperplane

    from sklearn import svm
    from sklearn.datasets import make_blobs

    # we create 40 separable points
    x, y = make_blobs(n_samples=10, centers=2, random_state=6)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(x, y)

    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
               edgecolors='k')
    plt.show()




# Using Naive Bayes



def naive():
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc = classification_report(y_test, y_pred)
    print(acc)
    acc1=accuracy_score(y_test,y_pred)*100
    print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
   # str = metrics.accuracy_score(y_test, y_pred) * 100
#    entry.delete(0, tk.END)
 #   entry.insert(0, str)


    window=Tk()
    window.geometry('300x300')
    window.title("Naive Bayes")
    frm=Frame(window,width=300,height=100)
    frm.pack(side=TOP)
    lbx = Label(frm,text="Accuracy of Naive Bayes",font=("Arial Bold",10)).grid(row=0,column=0)
    lby = Label(frm,text=acc1,font=("Arial Bold",10)).grid(row=0,column=1)
    window.mainloop()
    # Bar plot of Model

    import scikitplot as skplot
    skplot.metrics.plot_confusion_matrix(y_test, y_pred)
    p = metrics.classification_report(y_test, y_pred, output_dict=True)
    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.show()

def con1():
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # skplt.metrics.plot_confusion_matrix(y_test,y_pred)
    # plt.show()


    from sklearn.naive_bayes import GaussianNB
    from yellowbrick.classifier import ClassificationReport, ClassificationScoreVisualizer

    # Instantiate the classification model and visualizer
    bayes = GaussianNB()
    classes = ['Purchased', 'Not Purchased']
    visualizer = ClassificationReport(bayes, classes=classes, support=True)

    visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()
def PRC():
    # Precision recall curve Visualization

    from sklearn.metrics import precision_recall_curve ,average_precision_score

    from inspect import signature
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2 ,where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve for Naive Bayes: AP={0:0.2f}'.format(average_precision))

# Accuracy : 88.3%






#  Using KNN




def knn():
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=4)

    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc = classification_report(y_test, y_pred)
    print(acc)
    acc1=accuracy_score(y_test,y_pred)*100
    print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
    # import seaborn as sns
    # sns.heatmap(cm, annot=True, annot_kws={"size": 16})  # font size


    window=Tk()
    window.geometry('300x300')
    window.title("K Nearest Neighbour")
    frm=Frame(window,width=300,height=100)
    frm.pack(side=TOP)
    lbx = Label(frm,text="Accuracy of KNN",font=("Arial Bold",10)).grid(row=0,column=0)
    lby = Label(frm,text=acc1,font=("Arial Bold",10)).grid(row=0,column=1)
    window.mainloop()

    import scikitplot as skplot
    skplot.metrics.plot_confusion_matrix(y_test, y_pred)
    p = metrics.classification_report(y_test, y_pred, output_dict=True)
    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.show()

def con2():
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    skplt.metrics.plot_confusion_matrix(y_test ,y_pred)
    plt.show()
    # Precision Recall Report visulization

    from yellowbrick.classifier import ClassificationReport, ClassificationScoreVisualizer

    # Instantiate the classification model and visualizer
    knn = KNeighborsClassifier()
    classes = ['Purchased', 'Not Purchased']
    visualizer = ClassificationReport(knn, classes=classes, support=True)

    visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()

def PRCK():
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Precision recall report visulization

    from sklearn.metrics import precision_recall_curve, average_precision_score
    from inspect import signature

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve for KNN: AP={0:0.2f}'.format(average_precision))


# Accuracy : 85.0%







#  Using Logistic Regression
def logi():
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    acc = classification_report(y_test, y_pred)
    print(acc)
    acc1=accuracy_score(y_test,y_pred)
    print("Accuracy :", accuracy_score(y_test, y_pred) * 100)
    # import seaborn as sns
    # sns.heatmap(cm, annot=True, annot_kws={"size": 16})  # font size

   # str = metrics.accuracy_score(y_test, y_pred) * 100
#    entry.delete(0, tk.END)
 #   entry.insert(0, str)


    window=Tk()
    window.geometry('300x300')
    window.title("Logistic Regression")
    frm=Frame(window,width=300,height=100)
    frm.pack(side=TOP)
    lbx = Label(frm,text="Accuracy of Logistic Regression",font=("Arial Bold",10)).grid(row=0,column=0)
    lby = Label(frm,text=acc1,font=("Arial Bold",10)).grid(row=0,column=1)
    window.mainloop()


    # import seaborn as sns
    # sns.heatmap(cm, annot=True, annot_kws={"size": 15})  # font size
    import scikitplot as skplot
    skplot.metrics.plot_confusion_matrix(y_test, y_pred)
    p = metrics.classification_report(y_test, y_pred, output_dict=True)
    p1 = pd.DataFrame(p).transpose()
    p1 = p1.drop('support', 1)
    print(p1)
    p1.plot.bar()
    plt.show()


def con3():
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    skplt.metrics.plot_confusion_matrix(y_test ,y_pred)
    plt.show()

    # Precision Recall Report visulization

    from yellowbrick.classifier import ClassificationReport, ClassificationScoreVisualizer

    # Instantiate the classification model and visualizer
    lr = LogisticRegression()
    classes = ['Purchased', 'Not Purchased']
    visualizer = ClassificationReport(lr, classes=classes, support=True)

    visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
    visualizer.score(x_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()

def PRCL():
    # Precision recall report visulization

    import scikitplot as skplt
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    clf1 = LogisticRegression()
    clf1.fit(x_train, y_train)
    y_pred = clf1.predict(x_test)

    from sklearn.metrics import precision_recall_curve, average_precision_score
    from inspect import signature

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# Accuracy : 88.1%
# Comparison
def com():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        GaussianNB()]
    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    i = 0
    for clf in classifiers:
        clf.fit(x_train, y_train)
        name = clf.__class__.__name__
        print("=" * 30)
        print(name)
        print('****Results****')
        train_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))
        train_predictions = clf.predict_proba(x_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))
        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

        # plt.hist(log)
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_color_codes("muted")
        sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
        plt.xlabel('Accuracy %')
        plt.title('Classifier Accuracy')
        plt.show()




from sklearn.tree import DecisionTreeClassifier
mediaTree =DecisionTreeClassifier(criterion="entropy" ,max_depth=4)
print(mediaTree)
mediaTree.fit(x_train ,y_train)
predTree =mediaTree.predict(x_test)
print(predTree[0:5])
print(y_test[0:5])


##Evaluation
from sklearn import metrics
print("DecisionTrees's Accuracy:" ,metrics.accuracy_score(y_test ,predTree))
print("Confusion Matrix:" ,metrics.confusion_matrix(y_test ,predTree))
print("Classification Report:" ,metrics.classification_report(y_test ,predTree))

##Visualization
# import pydotplus
# import matplotlib.pyplot as plt
# from IPython.display import Image
# import matplotlib.image as mpimg

# from sklearn import tree

# filename="media.png"
# featureNames=df.columns[1:3]
# targetNames=df["Purchased"].unique().tolist()
# out=tree.export_graphviz(mediaTree,feature_names=featureNames,out_file=None,class_names=np.unique(y_train),filled=True,special_characters=True,rotate=False)
# graph=pydotplus.graph_from_dot_data(out)
# graph.write_png(filename)
# img=mpimg.imread('mediatree.png')
# imgplot=plt.imshow(img)
# plt.show()

from tkinter import *
window =Tk()
window.title("Classification Algorithms")
window.geometry('450x400')
window.configure(bg='pink')
lb1 =Label()
lb1.configure()
lb3 = Label(window, text=".......WELCOME TO WORLD OF CLASSIFICATION ALGORITHMS..........")
lb3.configure(bg='skyblue' ,fg='black')
lb3.grid(column=0, row=0)
lb2 = Label(window ,text="...........Click Here for Results........")
lb2.configure(bg='skyblue' ,fg='black')
lb2.grid(column= 0,row = 1)

#def clicked():
   # lb1.configure(naive())
def clickedN():
    lb1.configure(con1())
def clicked1():
    lb1.configure(knn())
def clickedK():
    lb1.configure(con2())
def clicked2():
    lb1.configure(logi())
def clickedL():
    lb1.configure(con3())
def clicked3():
    lb1.configure(svm())
def clickedS():
    lb1.configure(con4())
def clickedC():
    lb1.configure(com())
def clickedP():
    lb1.configure(PRC())


btn =Button(window, text="Naive Bayes", width=20, command=naive)
btn.grid(column=0, row=2, padx=10, pady=10)
btn.configure(bg="grey", fg="white")
btn = Button(window, text="PRN CLassification", width=20, command=clickedN)
btn.grid(column=1, row=2, padx=10, pady=10)
btn.configure(bg="grey", fg="white")

btn = Button(window, text="PR Curve", width=20, command=clickedP)
btn.grid(column=2, row=2, padx=10, pady=10)
btn.configure(bg="grey", fg="white")

btn = Button(window, text=" KNN", width=20, command=clicked1)
btn.grid(column=0, row=3, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

btn = Button(window, text=" PRK Classification ", width=20, command=clickedK)
btn.grid(column=1, row=3, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

btn = Button(window, text="PR Curve", width=20, command=PRCK)
btn.grid(column=2, row=3, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

btn = Button(window, text="Logistic", width=20, command=clicked2)
btn.grid(column=0, row=4, padx=50, pady=30)
btn.configure(bg="grey", fg="white")

btn = Button(window, text=" PRL Classification ", width=20, command=clickedL)
btn.grid(column=1, row=4, padx=50, pady=30)
btn.configure(bg="grey", fg="white")

btn = Button(window, text="PR Curve", width=20, command=PRCL)
btn.grid(column=2, row=4, padx=50, pady=30)
btn.configure(bg="grey", fg="white")

btn = Button(window, text="SVM ", width=20, command=clicked3)
btn.grid(column=0, row=5, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

btn = Button(window, text="PRS Classification", width=20, command=clickedS)
btn.grid(column=1, row=5, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

btn = Button(window, text="PR Curve", width=20, command=PRS)
btn.grid(column=2, row=5, padx=50, pady=30)
btn.configure(bg="sky blue", fg="black")

#btn_frame=Frame(window,width=800,height=700,bg="")
#btn_frame.pack()
#btn1 = Button(btn_frame,text="aaa",fg="white",width=120,height=3,bg="blue",bd=0,).grid(row=0,column=0,columnspan=4,padx=50,pady=30)
btn = Button(window, text="Hyperplan", fg="white", width=20, bg="grey", bd=0, command=HP).grid(column=0, row=6)
# btn.configure(bg="orange",fg="blue")

btn = Button(window, text="Comparison", width=20, command=clickedC)
btn.grid(column=1, row=6, padx=50, pady=30)
btn.configure(bg="grey", fg="white")

#root = tk.Tk()

#text = Text(root, width=10, height=5)

#text.insert(INSERT, "Accuracy")
#entry = tk.Entry(root, textvariable=str)
#text.pack()

#entry.pack(side=tk.RIGHT)

window.mainloop()


