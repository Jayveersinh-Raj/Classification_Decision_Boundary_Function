   
    import numpy as np
    import matplotlib.pyplot as plt

    # Plots the decision boundary created by a model predicting on X.
    def plot_decision_boundary(model, X, y):
    # Define the axis boundaries of the plot and create a meshgrid
      x_min, x_max = X[:, 0].min() - 0.1 , X[:, 0].max() + 0.1
      y_min, y_max = X[:, 0].min() - 0.1, X[:, 1].max() + 0.1
      # This are just the boundaries, the lowest and the highest

      # Read about numpy meshgrids to know more

      xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
      # Create X value (we are going to make predictions on these)
      x_in = np.c_[xx.ravel(), yy.ravel()] # Stack 2D arrays together

      # Make predictions
      y_pred = model.predict(x_in)

      # Check for multi-class
      if len(y_pred[0]) > 1:
        print("It is multiclass")
        # We will have to reshape our prediction to get them ready for plot
        y_pred = np.argmax(y_pred, axis = 1).reshape(xx.shape) 
        
      else:
        print("It is binary")
        y_pred = np.round(y_pred).reshape(xx.shape)
 
       # Plot the boundary
     plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
     plt.scatter(X[:, 0], X[:,1], c= y, cmap = plt.cm.RdYlBu)
     plt.xlim(xx.min(), xx.max())
     plt.ylim(yy.min(), yy.max())
