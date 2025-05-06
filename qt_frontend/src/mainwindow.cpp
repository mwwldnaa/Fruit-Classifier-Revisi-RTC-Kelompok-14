#include "mainwindow.h"
#include "plotwidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QGroupBox>
#include <QDoubleValidator>
#include <QIntValidator>
#include <QCoreApplication>
#include <QProgressBar>
#include <QTimer>
#include <QFile>
#include <QFont>
#include <QFrame>
#include <QFileDialog>

#ifdef _WIN32
    #define RUST_IMPORT __declspec(dllimport)
#else
    #define RUST_IMPORT
#endif

extern "C" {
    RUST_IMPORT bool train_network(
        const char* dataset_path,
        double** accuracies,
        double** losses,
        double* final_accuracy,
        size_t* length,
        size_t epochs
    );
    RUST_IMPORT char* predict(double weight, double size, double width, double height);
    RUST_IMPORT void free_array(double* ptr);
    RUST_IMPORT void free_string(char* ptr);
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    currentEpoch(0), totalEpochs(5000), finalAccuracy(0)
{
    setupUI();
    trainingTimer = new QTimer(this);
    connect(trainingTimer, &QTimer::timeout, this, &MainWindow::updateTrainingProgress);
}

void MainWindow::setupUI()
{
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setStyleSheet("background-color: #f5f5f5;");
    setCentralWidget(centralWidget);
    
    mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Header
    QLabel *titleLabel = new QLabel("Fruit Classifier Neural Network");
    titleLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #333;");
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);

    // Training Section
    QGroupBox *trainingGroup = new QGroupBox("Training Progress");
    trainingGroup->setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }"
                               "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }");
    QVBoxLayout *trainingLayout = new QVBoxLayout(trainingGroup);

    // Dataset selection
    QHBoxLayout *datasetLayout = new QHBoxLayout();
    selectDatasetButton = new QPushButton("Select Dataset");
    selectDatasetButton->setStyleSheet("QPushButton {"
                                     "background-color: #607d8b;"
                                     "border: none;"
                                     "color: white;"
                                     "padding: 8px;"
                                     "font-weight: bold;"
                                     "border-radius: 5px;"
                                     "}"
                                     "QPushButton:hover { background-color: #546e7a; }");
    datasetLabel = new QLabel("No dataset selected");
    datasetLabel->setStyleSheet("color: #666; font-size: 12px;");
    datasetLabel->setWordWrap(true);
    datasetLayout->addWidget(selectDatasetButton);
    datasetLayout->addWidget(datasetLabel, 1);
    trainingLayout->addLayout(datasetLayout);

    // Epoch input
    QHBoxLayout *epochLayout = new QHBoxLayout();
    QLabel *epochInputLabel = new QLabel("Epochs:");
    epochInputLabel->setStyleSheet("font-weight: bold;");
    epochInput = new QLineEdit();
    epochInput->setText("5000");
    epochInput->setValidator(new QIntValidator(100, 100000, this));
    epochInput->setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;");
    epochInput->setFixedWidth(100);
    epochLayout->addWidget(epochInputLabel);
    epochLayout->addWidget(epochInput);
    epochLayout->addStretch();
    trainingLayout->addLayout(epochLayout);

    trainButton = new QPushButton("Start Training");
    trainButton->setStyleSheet("QPushButton {"
                              "background-color: #4CAF50;"
                              "border: none;"
                              "color: white;"
                              "padding: 10px;"
                              "font-weight: bold;"
                              "border-radius: 5px;"
                              "}"
                              "QPushButton:hover { background-color: #45a049; }"
                              "QPushButton:disabled { background-color: #cccccc; }");
    trainingLayout->addWidget(trainButton);

    progressBar = new QProgressBar();
    progressBar->setRange(0, totalEpochs);
    progressBar->setTextVisible(true);
    progressBar->setStyleSheet("QProgressBar {"
                              "border: 1px solid #ddd;"
                              "border-radius: 5px;"
                              "text-align: center;"
                              "}"
                              "QProgressBar::chunk {"
                              "background-color: #4CAF50;"
                              "}");
    trainingLayout->addWidget(progressBar);

    epochLabel = new QLabel(QString("Epoch: 0/%1").arg(totalEpochs));
    epochLabel->setStyleSheet("font-size: 12px; color: #666;");
    trainingLayout->addWidget(epochLabel);

    accuracyLabel = new QLabel("Current Accuracy: 0%");
    accuracyLabel->setStyleSheet("font-size: 14px; font-weight: bold;");
    trainingLayout->addWidget(accuracyLabel);

    // Charts
    QHBoxLayout *chartLayout = new QHBoxLayout();
    
    accuracyPlot = new PlotWidget();
    accuracyPlot->setMinimumSize(400, 250);
    accuracyPlot->setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 5px;");
    accuracyPlot->setTitle("Training Accuracy");
    accuracyPlot->setPlotColor(QColor(65, 105, 225));
    accuracyPlot->setYRange(0.0, 1.0);
    chartLayout->addWidget(accuracyPlot);
    
    lossPlot = new PlotWidget();
    lossPlot->setMinimumSize(400, 250);
    lossPlot->setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 5px;");
    lossPlot->setTitle("Training Loss");
    lossPlot->setPlotColor(QColor(220, 20, 60));
    lossPlot->setYRange(0.0, 2.0);
    chartLayout->addWidget(lossPlot);
    
    trainingLayout->addLayout(chartLayout);
    mainLayout->addWidget(trainingGroup);

    // Prediction Section
    QGroupBox *predictionGroup = new QGroupBox("Fruit Prediction");
    predictionGroup->setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }"
                                 "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }");
    QVBoxLayout *predictionLayout = new QVBoxLayout(predictionGroup);

    QGridLayout *inputLayout = new QGridLayout();
    const QStringList featureNames = {"Weight (g)", "Size (cm)", "Width (cm)", "Height (cm)"};
    for (int i = 0; i < featureNames.size(); ++i) {
        QLabel *label = new QLabel(featureNames[i]);
        label->setStyleSheet("font-weight: bold;");
        inputLayout->addWidget(label, i, 0);
        
        QLineEdit *input = new QLineEdit();
        input->setPlaceholderText(featureNames[i]);
        input->setValidator(new QDoubleValidator(0.1, 10000.0, 2, input));
        input->setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 3px;");
        inputLayout->addWidget(input, i, 1);
        featureInputs.append(input);
    }
    predictionLayout->addLayout(inputLayout);

    predictButton = new QPushButton("Predict Fruit");
    predictButton->setStyleSheet("QPushButton {"
                               "background-color: #2196F3;"
                               "border: none;"
                               "color: white;"
                               "padding: 10px;"
                               "font-weight: bold;"
                               "border-radius: 5px;"
                               "}"
                               "QPushButton:hover { background-color: #0b7dda; }");
    predictionLayout->addWidget(predictButton);

    predictionLabel = new QLabel("Prediction: -");
    predictionLabel->setAlignment(Qt::AlignCenter);
    predictionLabel->setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;");
    
    QFrame *predictionFrame = new QFrame();
    predictionFrame->setFrameShape(QFrame::Box);
    predictionFrame->setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 5px;");
    QVBoxLayout *frameLayout = new QVBoxLayout(predictionFrame);
    frameLayout->addWidget(predictionLabel);
    
    predictionLayout->addWidget(predictionFrame);
    mainLayout->addWidget(predictionGroup);

    connect(selectDatasetButton, &QPushButton::clicked, this, &MainWindow::selectDataset);
    connect(trainButton, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(predictButton, &QPushButton::clicked, this, &MainWindow::predictFruit);
}

void MainWindow::selectDataset()
{
    QString path = QFileDialog::getOpenFileName(this, "Select Dataset File", "", "CSV Files (*.csv)");
    if (!path.isEmpty()) {
        datasetPath = path;
        datasetLabel->setText(QFileInfo(path).fileName());
        datasetLabel->setToolTip(path);
    }
}

void MainWindow::startTraining()
{
    if (datasetPath.isEmpty()) {
        QMessageBox::critical(this, "Error", "Please select a dataset file first");
        return;
    }

    trainButton->setEnabled(false);
    currentEpoch = 0;
    accuracies.clear();
    losses.clear();
    
    totalEpochs = epochInput->text().toInt();
    progressBar->setRange(0, totalEpochs);
    epochLabel->setText(QString("Epoch: 0/%1").arg(totalEpochs));
    
    if (!QFile::exists(datasetPath)) {
        QMessageBox::critical(this, "Error", "Dataset file not found at:\n" + datasetPath);
        trainButton->setEnabled(true);
        return;
    }

    // Call Rust training function
    double* rustAccuracies = nullptr;
    double* rustLosses = nullptr;
    double rustFinalAccuracy = 0.0;
    size_t dataLength = 0;
    
    bool success = train_network(
        datasetPath.toUtf8().constData(),
        &rustAccuracies,
        &rustLosses,
        &rustFinalAccuracy,
        &dataLength,
        static_cast<size_t>(totalEpochs)
    );

    if (!success) {
        QMessageBox::critical(this, "Error", "Training failed");
        trainButton->setEnabled(true);
        return;
    }

    // Copy data from Rust
    accuracies.resize(dataLength);
    losses.resize(dataLength);
    for (size_t i = 0; i < dataLength; ++i) {
        accuracies[i] = rustAccuracies[i];
        losses[i] = rustLosses[i];
    }

    finalAccuracy = rustFinalAccuracy;
    
    // Free Rust-allocated memory
    free_array(rustAccuracies);
    free_array(rustLosses);

    // Start timer for animation
    trainingTimer->start(10); // Faster update for smoother animation
}

void MainWindow::updateTrainingProgress()
{
    if (currentEpoch >= accuracies.size()) {
        trainingTimer->stop();
        trainButton->setEnabled(true);
        accuracyLabel->setText(QString("Final Accuracy: %1%").arg(finalAccuracy * 100, 0, 'f', 2));
        return;
    }

    progressBar->setValue(currentEpoch + 1);
    epochLabel->setText(QString("Epoch: %1/%2").arg(currentEpoch + 1).arg(totalEpochs));
    
    // Update accuracy label
    double currentAccuracy = accuracies[currentEpoch];
    accuracyLabel->setText(QString("Current Accuracy: %1%").arg(currentAccuracy * 100, 0, 'f', 2));
    
    // Update charts with data up to current epoch
    QVector<double> partialAccuracies(accuracies.begin(), accuracies.begin() + currentEpoch + 1);
    QVector<double> partialLosses(losses.begin(), losses.begin() + currentEpoch + 1);
    
    accuracyPlot->setData(partialAccuracies);
    lossPlot->setData(partialLosses);
    
    currentEpoch++;
}

void MainWindow::predictFruit()
{
    bool ok;
    double weight = featureInputs[0]->text().toDouble(&ok);
    if (!ok || weight <= 0) {
        predictionLabel->setText("Invalid weight (must be > 0)");
        predictionLabel->setStyleSheet("color: #d32f2f;");
        return;
    }
    
    double size = featureInputs[1]->text().toDouble(&ok);
    if (!ok || size <= 0) {
        predictionLabel->setText("Invalid size (must be > 0)");
        predictionLabel->setStyleSheet("color: #d32f2f;");
        return;
    }
    
    double width = featureInputs[2]->text().toDouble(&ok);
    if (!ok || width <= 0) {
        predictionLabel->setText("Invalid width (must be > 0)");
        predictionLabel->setStyleSheet("color: #d32f2f;");
        return;
    }
    
    double height = featureInputs[3]->text().toDouble(&ok);
    if (!ok || height <= 0) {
        predictionLabel->setText("Invalid height (must be > 0)");
        predictionLabel->setStyleSheet("color: #d32f2f;");
        return;
    }

    char* result = predict(weight, size, width, height);
    QString prediction = QString::fromUtf8(result);
    free_string(result);
    
    QString color;
    if (prediction == "unknown") {
        color = "#ff9800";
    } else {
        color = "#388e3c";
    }
    
    predictionLabel->setText(QString("Prediction: %1").arg(prediction));
    predictionLabel->setStyleSheet(QString("color: %1;").arg(color));
}

MainWindow::~MainWindow()
{
    delete trainingTimer;
}