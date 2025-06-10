import React, { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Chart = ({
  title,
  data,
  height = 300,
  showLegend = true,
  showGrid = true,
  yAxisMin = 0,
  yAxisMax = 100,
  yAxisUnit = '%',
  theme = 'light',
}) => {
  const chartRef = useRef(null);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: showLegend,
        position: 'top',
        labels: {
          color: theme === 'dark' ? '#FFFFFF' : '#000000',
        },
      },
      title: {
        display: true,
        text: title,
        color: theme === 'dark' ? '#FFFFFF' : '#000000',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
    },
    scales: {
      x: {
        grid: {
          display: showGrid,
          color: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          color: theme === 'dark' ? '#FFFFFF' : '#000000',
        },
      },
      y: {
        min: yAxisMin,
        max: yAxisMax,
        grid: {
          display: showGrid,
          color: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          color: theme === 'dark' ? '#FFFFFF' : '#000000',
          callback: (value) => `${value}${yAxisUnit}`,
        },
      },
    },
  };

  return (
    <div style={{ height, width: '100%', position: 'relative' }}>
      <Line ref={chartRef} data={data} options={options} />
    </div>
  );
};

export default Chart; 