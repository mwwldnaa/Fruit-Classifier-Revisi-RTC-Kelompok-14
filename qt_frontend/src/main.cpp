#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    // Set application style and font
    QApplication::setStyle("Fusion");
    QFont font;
    font.setFamily("Segoe UI");
    font.setPointSize(10);
    QApplication::setFont(font);
    
    MainWindow w;
    w.setWindowTitle("Fruit Classifier");
    w.resize(900, 800);
    w.show();
    
    return a.exec();
}