#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <QWidget>
#include <QVector>
#include <QString>
#include <QColor>

class PlotWidget : public QWidget {
    Q_OBJECT
public:
    explicit PlotWidget(QWidget *parent = nullptr);
    void setData(const QVector<double>& data);
    void setPlotColor(const QColor &color);
    void setYRange(double minY, double maxY);
    void setTitle(const QString &title);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QVector<double> m_data;
    QColor m_plotColor;
    double m_minY;
    double m_maxY;
    QString m_title;
};

#endif // PLOTWIDGET_H