import numpy as np

def escalador_padrao(data, f_media = True, f_std = True):
    '''
    ********************************************************************
    Data de Criacao     : 09/05/2020
    Data de modificacao : 00/00/0000
    --------------------------------------------------------------------
    escalador_padrao : escalar as colunas
    --------------------------------------------------------------------
    Paramentros
    --------------------------------------------------------------------
    data: valores no formato de matiz numpy
    f_media: usa a media (default=True)
    f_std:  usa o desvio padrao (default=True)
    --------------------------------------------------------------------
    Retorno
    --------------------------------------------------------------------
    z: data escalado
    --------------------------------------------------------------------
    OBS:
    --------------------------------------------------------------------
    z[:,0] = (data[:,0] - data[:,0].media())/data[:,0].std()
    z[:,1] = (data[:,1] - data[:,1].media())/data[:,1].std()
    z[:,2] = (data[:,2] - data[:,2].media())/data[:,2].std()
    ...
    z[:,n_col] = (data[:,2] - data[:,2].media())/data[:,2].std()
    ********************************************************************
    '''
    n_lin, n_col = data.shape

    z = np.empty((n_col, n_lin),dtype=np.float64)

    (mean, std) = 0.0e0, 1.0e0
    for j, col in enumerate(data.transpose()):
        if(f_media):
            mean = col.mean()
        if(f_std):
            std  = col.std()

        z[j,:] = (col - mean)/std

    return z.transpose()

def escalador_minmax(data, f_range=(0, 1)):
    '''
    ********************************************************************
    Data de Criacao     : 12/05/2020
    Data de modificacao : 00/00/0000
    --------------------------------------------------------------------
    escalador_padrao : escalar as colunas min max
    --------------------------------------------------------------------
    Paramentros
    --------------------------------------------------------------------
    data: valores no formato de matiz numpy
    --------------------------------------------------------------------
    Retorno
    --------------------------------------------------------------------
    z: data escalado
    --------------------------------------------------------------------
    OBS:
    --------------------------------------------------------------------
    z[:,0] = (d[:,0] - d[:,0].min())/(d[:,0].max() - d[:,0].min())
    z[:,1] = (d[:,1] - d[:,1].min())/(d[:,1].max() - d[:,1].min())
    z[:,2] = (d[:,2] - d[:,2].min())/(d[:,2].max() - d[:,2].min())
    ...
    ********************************************************************
    '''
    n_lin, n_col = data.shape
    a, b = f_range
    z = np.empty((n_col, n_lin),dtype=np.float64)

    for j, col in enumerate(data.transpose()):
        (x_min, x_max) = col.min(), col.max()
        z[j,:] = (col - x_min)/(x_max - x_min)

    return z.transpose() *(b - a) + a
