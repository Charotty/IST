import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import * as Progress from '@radix-ui/react-progress';

interface SignalPanelProps {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  predictedChange: number;
  currentModel: string;
}

export function SignalPanel({ signal, confidence, predictedChange, currentModel }: SignalPanelProps) {
  const getSignalColor = () => {
    switch (signal) {
      case 'BUY':
        return '#22C55E';
      case 'SELL':
        return '#EF4444';
      default:
        return '#9CA3AF';
    }
  };

  const getSignalIcon = () => {
    switch (signal) {
      case 'BUY':
        return <TrendingUp size={24} />;
      case 'SELL':
        return <TrendingDown size={24} />;
      default:
        return <Minus size={24} />;
    }
  };

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#161B22', width: '280px' }}>
      {/* Current Signal */}
      <div className="p-6 border-b" style={{ borderColor: '#1F2933' }}>
        <div className="text-xs mb-3" style={{ color: '#9CA3AF' }}>
          Current Signal
        </div>
        <div
          className="flex items-center gap-3 p-4 rounded-lg"
          style={{ backgroundColor: '#1F2933' }}
        >
          <div style={{ color: getSignalColor() }}>{getSignalIcon()}</div>
          <div className="text-2xl" style={{ color: getSignalColor() }}>
            {signal}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="p-6 space-y-6 border-b" style={{ borderColor: '#1F2933' }}>
        {/* Predicted Change */}
        <div>
          <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
            Predicted Change (ΔPₜ)
          </div>
          <div
            className="text-xl"
            style={{
              color: predictedChange >= 0 ? '#22C55E' : '#EF4444',
            }}
          >
            {predictedChange >= 0 ? '+' : ''}
            {predictedChange.toFixed(3)}%
          </div>
        </div>

        {/* Confidence */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs" style={{ color: '#9CA3AF' }}>
              Confidence (p̂ₜ)
            </div>
            <div className="text-sm" style={{ color: '#C9D1D9' }}>
              {(confidence * 100).toFixed(1)}%
            </div>
          </div>
          <Progress.Root
            className="relative overflow-hidden rounded-full w-full"
            style={{
              backgroundColor: '#1F2933',
              height: '8px',
            }}
            value={confidence * 100}
          >
            <Progress.Indicator
              className="h-full transition-transform duration-300 ease-in-out"
              style={{
                backgroundColor: confidence > 0.7 ? '#22C55E' : confidence > 0.4 ? '#F59E0B' : '#9CA3AF',
                transform: `translateX(-${100 - confidence * 100}%)`,
              }}
            />
          </Progress.Root>
        </div>

        {/* Active Model */}
        <div>
          <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
            Active Model
          </div>
          <div
            className="px-3 py-2 rounded text-center"
            style={{
              backgroundColor: '#3B82F6',
              color: '#FFFFFF',
            }}
          >
            {currentModel}
          </div>
        </div>
      </div>

      {/* Risk Warning */}
      {confidence < 0.5 && (
        <div className="p-4 m-4 rounded" style={{ backgroundColor: '#F59E0B20', borderLeft: '3px solid #F59E0B' }}>
          <div className="text-xs" style={{ color: '#F59E0B' }}>
            Low confidence signal. Consider manual review.
          </div>
        </div>
      )}
    </div>
  );
}
