#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>

class QLabel;
class QLineEdit;
class QPushButton;
class QVBoxLayout;
class PlotWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void trainModel();
    void predictFruit();

private:
    PlotWidget *accuracyPlot;
    PlotWidget *lossPlot;
    QVector<QLineEdit*> featureInputs;
    QLabel *predictionLabel;
    QLabel *accuracyLabel;
    QPushButton *trainButton;
    QVBoxLayout *mainLayout;
};

#endif // MAINWINDOW_H