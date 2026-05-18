# from sklearn import tree
# x = [[0,0],[1,1]]
# y = [0,1]

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x,y)

# print(clf.predict([[2.,2.]]))


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit([[0, 0], [1, 1]], [0, 1])
print(model.predict([[2.,2.]]))