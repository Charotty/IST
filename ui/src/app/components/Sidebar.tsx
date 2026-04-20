import { Play, Square, Activity } from 'lucide-react';
import { useState } from 'react';

interface SidebarProps {
  onPairSelect: (pair: string) => void;
  selectedPair: string;
  systemStatus: 'RUNNING' | 'STOPPED' | 'ERROR';
  onStart: () => void;
  onStop: () => void;
}

const TRADING_PAIRS = [
  'BTC/USDT',
  'ETH/USDT',
  'BNB/USDT',
  'SOL/USDT',
  'XRP/USDT',
  'ADA/USDT',
];

export function Sidebar({ onPairSelect, selectedPair, systemStatus, onStart, onStop }: SidebarProps) {
  const [mode, setMode] = useState<'paper' | 'live'>('paper');

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#161B22', width: '200px' }}>
      {/* Status Indicator */}
      <div className="p-4 border-b" style={{ borderColor: '#1F2933' }}>
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor:
                systemStatus === 'RUNNING'
                  ? '#22C55E'
                  : systemStatus === 'ERROR'
                  ? '#EF4444'
                  : '#9CA3AF',
            }}
          />
          <span className="text-xs" style={{ color: '#C9D1D9' }}>
            {systemStatus}
          </span>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="p-4 space-y-2 border-b" style={{ borderColor: '#1F2933' }}>
        <button
          onClick={onStart}
          disabled={systemStatus === 'RUNNING'}
          className="w-full px-3 py-2 rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          style={{
            backgroundColor: systemStatus === 'RUNNING' ? '#1F2933' : '#22C55E',
            color: '#FFFFFF',
          }}
        >
          <Play size={14} />
          <span className="text-xs">Start</span>
        </button>
        <button
          onClick={onStop}
          disabled={systemStatus === 'STOPPED'}
          className="w-full px-3 py-2 rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          style={{
            backgroundColor: systemStatus === 'STOPPED' ? '#1F2933' : '#EF4444',
            color: '#FFFFFF',
          }}
        >
          <Square size={14} />
          <span className="text-xs">Stop</span>
        </button>
      </div>

      {/* Mode Toggle */}
      <div className="p-4 border-b" style={{ borderColor: '#1F2933' }}>
        <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
          Mode
        </div>
        <div className="flex gap-1 rounded" style={{ backgroundColor: '#0D1117', padding: '2px' }}>
          <button
            onClick={() => setMode('paper')}
            className="flex-1 px-2 py-1 rounded text-xs transition-colors"
            style={{
              backgroundColor: mode === 'paper' ? '#3B82F6' : 'transparent',
              color: mode === 'paper' ? '#FFFFFF' : '#9CA3AF',
            }}
          >
            Paper
          </button>
          <button
            onClick={() => setMode('live')}
            className="flex-1 px-2 py-1 rounded text-xs transition-colors"
            style={{
              backgroundColor: mode === 'live' ? '#3B82F6' : 'transparent',
              color: mode === 'live' ? '#FFFFFF' : '#9CA3AF',
            }}
          >
            Live
          </button>
        </div>
      </div>

      {/* Trading Pairs */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
            Trading Pairs
          </div>
          <div className="space-y-1">
            {TRADING_PAIRS.map((pair) => (
              <button
                key={pair}
                onClick={() => onPairSelect(pair)}
                className="w-full px-3 py-2 rounded text-left transition-colors"
                style={{
                  backgroundColor: selectedPair === pair ? '#3B82F6' : 'transparent',
                  color: selectedPair === pair ? '#FFFFFF' : '#C9D1D9',
                }}
              >
                <div className="text-xs">{pair}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
