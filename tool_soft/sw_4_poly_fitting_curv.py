def get_linear_fitted_graph(arr, deg=4):
    # arr = np.array([[x1, y1], [x2, y2], ...])
    X2 = [[k**n for n in range(1, deg)] for k in arr[:, 0]]
    reg2 = linear_model.LinearRegression()
    reg2.fit(X2,arr[:, 1])
    print(reg2.intercept_)
    print(reg2.coef_)

    xp2 = [k for k in range(int(np.min(arr[:, 0])), int(np.max(arr[:, 0]))+1)]
    Xp2 = [[xp2[k]**n for n in range(1, deg)] for k in range(len(xp2))]
    yp2 = reg2.predict(Xp2)
    plt.figure(figsize=(3,3))
    # plt.xlim(0,10)
    # plt.ylim(0,30)
    plt.plot(arr[:, 0], arr[:, 1], "x-")
    plt.plot(xp2, yp2)
