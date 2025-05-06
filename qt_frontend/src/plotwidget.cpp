// plotwidget.cpp
#include "plotwidget.h"
#include <QPainter>
#include <QPen>
#include <QPainterPath>
#include <algorithm>

PlotWidget::PlotWidget(QWidget *parent) : QWidget(parent), 
    m_plotColor(Qt::blue), 
    m_minY(0.0), 
    m_maxY(1.0),
    m_gridColor(QColor(200, 200, 200)),
    m_bgColor(Qt::white)
{
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

void PlotWidget::setData(const QVector<double> &data) {
    m_data = data;
    if (!m_data.isEmpty()) {
        auto minmax = std::minmax_element(m_data.begin(), m_data.end());
        m_minY = *minmax.first;
        m_maxY = *minmax.second;
        
        // Add some padding and ensure reasonable ranges
        double range = m_maxY - m_minY;
        if (range < 0.1) {
            range = 0.1;
            m_minY = qMax(0.0, m_minY - 0.05);
            m_maxY = m_minY + range;
        } else {
            m_minY = qMax(0.0, m_minY - 0.1 * range);
            m_maxY = m_maxY + 0.1 * range;
        }
    }
    update();
}

void PlotWidget::setPlotColor(const QColor &color) {
    m_plotColor = color;
    update();
}

void PlotWidget::setYRange(double minY, double maxY) {
    m_minY = minY;
    m_maxY = maxY;
    update();
}

void PlotWidget::setTitle(const QString &title) {
    m_title = title;
    update();
}

void PlotWidget::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw background
    painter.fillRect(rect(), m_bgColor);
    
    // Draw title
    if (!m_title.isEmpty()) {
        painter.setPen(Qt::black);
        QFont font = painter.font();
        font.setBold(true);
        font.setPointSize(10);
        painter.setFont(font);
        painter.drawText(rect().adjusted(10, 10, -10, -10), Qt::AlignTop | Qt::AlignLeft, m_title);
    }
    
    // Draw grid
    painter.setPen(QPen(m_gridColor, 1, Qt::DotLine));
    int xSteps = 10;
    int ySteps = 10;
    
    for (int i = 0; i <= xSteps; i++) {
        int x = i * width() / xSteps;
        painter.drawLine(x, 0, x, height());
    }
    
    for (int i = 0; i <= ySteps; i++) {
        int y = i * height() / ySteps;
        painter.drawLine(0, y, width(), y);
        
        // Draw Y axis labels
        if (i > 0 && i < ySteps) {
            double value = m_maxY - (i * (m_maxY - m_minY) / ySteps);
            painter.drawText(QRect(5, y - 10, 50, 20), 
                           Qt::AlignLeft | Qt::AlignVCenter, 
                           QString::number(value, 'f', 2));
        }
    }

    if (m_data.isEmpty()) return;

    // Draw axes
    painter.setPen(QPen(Qt::black, 2));
    painter.drawLine(0, height()-1, width(), height()-1); // X axis
    painter.drawLine(0, 0, 0, height()); // Y axis

    // Draw curve with gradient
    QLinearGradient gradient(0, 0, 0, height());
    gradient.setColorAt(0, m_plotColor.lighter(120));
    gradient.setColorAt(1, m_plotColor.darker(120));
    
    QPen curvePen(QBrush(gradient), 3);
    curvePen.setCapStyle(Qt::RoundCap);
    curvePen.setJoinStyle(Qt::RoundJoin);
    painter.setPen(curvePen);

    QPainterPath path;
    for (int i = 0; i < m_data.size(); ++i) {
        double x = i * width() / double(m_data.size() - 1);
        double y = height() - ((m_data[i] - m_minY) / (m_maxY - m_minY)) * height();
        y = qMax(0.0, qMin(double(height()), y));

        if (i == 0) {
            path.moveTo(x, y);
        } else {
            path.lineTo(x, y);
        }
    }
    painter.drawPath(path);

    // Draw X axis labels
    painter.setPen(Qt::black);
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);

    painter.drawText(QRect(0, height() - 20, 30, 20), 
                   Qt::AlignLeft | Qt::AlignVCenter, "0");
    painter.drawText(QRect(width() - 30, height() - 20, 30, 20), 
                   Qt::AlignRight | Qt::AlignVCenter, QString::number(m_data.size() - 1));
}