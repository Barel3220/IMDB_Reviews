import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.f_scores = []
        self.g_means = []

    def log_epoch(self, epoch, loss, accuracy, f_score, g_mean):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.f_scores.append(f_score)
        self.g_means.append(g_mean)

    def plot_metrics(self, filename):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot Loss
        axs[0, 0].plot(self.epochs, self.losses, label='Loss')
        axs[0, 0].set_title('Loss over Epochs')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot Accuracy
        axs[0, 1].plot(self.epochs, self.accuracies, label='Accuracy', color='g')
        axs[0, 1].set_title('Accuracy over Epochs')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot F-score
        axs[1, 0].plot(self.epochs, self.f_scores, label='F-score', color='r')
        axs[1, 0].set_title('F-score over Epochs')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('F-score')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot G-mean
        axs[1, 1].plot(self.epochs, self.g_means, label='G-mean', color='m')
        axs[1, 1].set_title('G-mean over Epochs')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('G-mean')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
