import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';

interface MainChartProps {
  pair: string;
}

export function MainChart({ pair }: MainChartProps) {
  const chartData = useMemo(() => {
    const data = [];
    const basePrice = pair.startsWith('BTC') ? 45000 : pair.startsWith('ETH') ? 2500 : 300;

    for (let i = 0; i < 50; i++) {
      const volatility = basePrice * 0.02;
      const open = basePrice + (Math.random() - 0.5) * volatility;
      const close = open + (Math.random() - 0.5) * volatility;
      const high = Math.max(open, close) + Math.random() * volatility * 0.5;
      const low = Math.min(open, close) - Math.random() * volatility * 0.5;

      data.push({
        time: new Date(Date.now() - (49 - i) * 60000).toLocaleTimeString('ru-RU', {
          hour: '2-digit',
          minute: '2-digit',
        }),
        open,
        high,
        low,
        close,
        prediction: close + (Math.random() - 0.5) * volatility * 0.3,
        signal: i === 30 ? 'BUY' : i === 45 ? 'SELL' : null,
      });
    }
    return data;
  }, [pair]);

  const currentPrice = chartData[chartData.length - 1]?.close || 0;
  const priceChange = ((currentPrice - chartData[0]?.close) / chartData[0]?.close) * 100;

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#0D1117' }}>
      {/* Header */}
      <div className="p-4 border-b" style={{ borderColor: '#1F2933' }}>
        <div className="flex items-baseline gap-4">
          <h2 className="text-xl" style={{ color: '#C9D1D9' }}>
            {pair}
          </h2>
          <div className="text-2xl" style={{ color: '#FFFFFF' }}>
            ${currentPrice.toFixed(2)}
          </div>
          <div
            className="text-sm"
            style={{
              color: priceChange >= 0 ? '#22C55E' : '#EF4444',
            }}
          >
            {priceChange >= 0 ? '+' : ''}
            {priceChange.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 p-4" style={{ minHeight: '400px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2933" />
            <XAxis
              dataKey="time"
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
              interval={9}
            />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2933',
                border: 'none',
                borderRadius: '4px',
                color: '#C9D1D9',
              }}
              labelStyle={{ color: '#9CA3AF' }}
            />

            {/* Close price line */}
            <Line
              type="monotone"
              dataKey="close"
              stroke="#22C55E"
              strokeWidth={2}
              dot={false}
              name="Price"
              id="price-line"
            />

            {/* Prediction line */}
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
              strokeDasharray="5 5"
              name="Prediction"
              id="prediction-line"
            />

            {/* Buy signal */}
            {chartData.find((entry) => entry.signal === 'BUY') && (
              <ReferenceDot
                key="buy-signal"
                x={chartData.find((entry) => entry.signal === 'BUY')?.time}
                y={chartData.find((entry) => entry.signal === 'BUY')?.low}
                r={6}
                fill="#22C55E"
                stroke="#FFFFFF"
                strokeWidth={2}
                isFront
              />
            )}

            {/* Sell signal */}
            {chartData.find((entry) => entry.signal === 'SELL') && (
              <ReferenceDot
                key="sell-signal"
                x={chartData.find((entry) => entry.signal === 'SELL')?.time}
                y={chartData.find((entry) => entry.signal === 'SELL')?.high}
                r={6}
                fill="#EF4444"
                stroke="#FFFFFF"
                strokeWidth={2}
                isFront
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
