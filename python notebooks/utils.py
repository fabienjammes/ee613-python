def plot_data_2D(x, y_true, y_pred, colors = ['r', 'k', 'b'], title = 'Linear regression', alphas = None):
    if alphas is None:
        alphas = np.ones(len(x))
    #plot the predicted data
    plt.plot(x,y_pred, '-' + colors[0])
    
    #plot the error bar and the true data
    for i in range(len(x)):
        #plot the true data
        plt.plot(x[i],y_true[i], '.' + colors[1])#,alpha = alphas[i])
        plt.plot([x[i],x[i]], [y_true[i], y_pred[i]], '-'+colors[2], alpha = alphas[i])
    plt.xlabel(u'x\u2081')
    plt.ylabel(u'y\u2081')
    
    plt.title(title)
    
    return