#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QTimer>

class QLabel;
class QLineEdit;
class QPushButton;
class QVBoxLayout;
class QHBoxLayout;
class PlotWidget;
class QProgressBar;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void selectDataset();
    void startTraining();
    void updateTrainingProgress();
    void predictFruit();

private:
    void setupUI();
    void setupTrainingChart();
    void updateChart();

    PlotWidget *accuracyPlot;
    PlotWidget *lossPlot;
    QVector<QLineEdit*> featureInputs;
    QLabel *predictionLabel;
    QLabel *accuracyLabel;
    QLabel *epochLabel;
    QLabel *datasetLabel;
    QPushButton *trainButton;
    QPushButton *predictButton;
    QPushButton *selectDatasetButton;
    QProgressBar *progressBar;
    QVBoxLayout *mainLayout;
    QLineEdit *epochInput;
    
    QTimer *trainingTimer;
    int currentEpoch;
    int totalEpochs;
    QVector<double> accuracies;
    QVector<double> losses;
    double finalAccuracy;
    QString datasetPath;
};

#endif // MAINWINDOW_H