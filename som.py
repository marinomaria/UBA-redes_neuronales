import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SOM:
    def __init__(self, featureCount, mapSize, seed = 3):
        self.rng = np.random.default_rng(seed)
        self.N = featureCount
        self.mapSize = mapSize
        self.coord_map = np.array([[(i,j) for j in range(self.mapSize)] for i in range(self.mapSize)])
        self.W = self.rng.normal(0, 1, (self.mapSize, self.mapSize, self.N))


    def best_matching_unit(self, x):
        e = x - self.W
        n = np.linalg.norm(e, axis = 2)
        bmu = np.unravel_index(np.argmin(n), n.shape)

        return bmu
    
    def train(self, X: np.array, lr_start: float, lr_decay: float, ir_start: float, ir_decay: float, epochs: int, lr_function_decay = lambda x: x, ir_function_decay = lambda x: x):
        I = X.shape[0]
        for epoch in range(epochs):
            for i in range(I):
                
                x = X[i].reshape(1, self.N)
                e = x - self.W
                bmu = self.best_matching_unit(x)

                d = np.linalg.norm(self.coord_map - bmu, axis = 2)

                lr = lr_start * np.exp(- lr_function_decay(epoch) * lr_decay)
                ir = ir_start * np.exp(- ir_function_decay(epoch) * ir_decay)
                pf = np.exp(-d / (2 * np.square(ir))).reshape((self.mapSize, self.mapSize, 1))
                dW = lr * pf * e
                self.W += dW

            if epoch % 100 == 0: print(f'Epoch {epoch} of {epochs}')
    
    def weight_matrix(self):
        return self.W
    
    def plotMap(self, X: np.array, labels: list[int]):
        I = X.shape[0]
        map = np.empty(shape=(self.mapSize, self.mapSize), dtype=object)

        for row in range(self.mapSize):
            for col in range(self.mapSize):
                map[row][col] = []

        for i in range(I):
            x = X[i].reshape(1, self.N)
            bmu = self.best_matching_unit(x)
            map[bmu[0]][bmu[1]].append(labels[i]) 
        
        label_map = np.zeros(shape=(self.mapSize, self.mapSize),dtype=np.int64)
        for row in range(self.mapSize):
            for col in range(self.mapSize):
                label_list = map[row][col]
                if len(label_list) == 0:
                    label = 0 # Con este número identificamos a las unidades sin activar
                else:
                    label = max(label_list, key = label_list.count)
                    label_map[row][col] = label

        # whitesmoke representa las unidades no activadas
        cmap = sns.color_palette(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan','whitesmoke'])
        sns.heatmap(label_map, cmap=cmap).invert_yaxis()
        plt.show()

    def plotScatter(self, X: np.array, labels: list[int]):
        I = X.shape[0]

        bmu_x = []
        bmu_y = []
        for i in range(I):
            x = X[i].reshape(1, self.N)
            bmu = self.best_matching_unit(x)
            bmu_x.append(bmu[0]) 
            bmu_y.append(bmu[1]) 

        # Grafico cada instancia como un punto dentro de la unidad de salida correspondiente, agrego una distribución normal para no tener todos los puntos en el centro de la unidad
        plt.scatter(bmu_x + np.random.normal(0, 0.15, I), bmu_y + np.random.normal(0, 0.15, I), c = labels, cmap = 'tab10', alpha = 0.5)
        plt.title("Detalle por instancia")
        plt.show()