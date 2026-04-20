import * as Tabs from '@radix-ui/react-tabs';
import { Activity, DollarSign, BarChart3 } from 'lucide-react';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  message: string;
}

interface Trade {
  id: string;
  timestamp: string;
  pair: string;
  type: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice?: number;
  pnl?: number;
}

interface BottomPanelProps {
  logs: LogEntry[];
  trades: Trade[];
  metrics: {
    totalPnL: number;
    winRate: number;
    totalTrades: number;
    maxDrawdown: number;
  };
}

export function BottomPanel({ logs, trades, metrics }: BottomPanelProps) {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error':
        return '#EF4444';
      case 'warning':
        return '#F59E0B';
      default:
        return '#9CA3AF';
    }
  };

  return (
    <div className="h-full" style={{ backgroundColor: '#0D1117' }}>
      <Tabs.Root defaultValue="logs" className="h-full flex flex-col">
        <Tabs.List
          className="flex border-b"
          style={{ borderColor: '#1F2933', backgroundColor: '#161B22' }}
        >
          <Tabs.Trigger
            value="logs"
            className="px-4 py-3 text-sm flex items-center gap-2 transition-colors data-[state=active]:border-b-2"
            style={{
              color: '#C9D1D9',
              borderColor: '#3B82F6',
            }}
          >
            <Activity size={14} />
            Logs
          </Tabs.Trigger>
          <Tabs.Trigger
            value="trades"
            className="px-4 py-3 text-sm flex items-center gap-2 transition-colors data-[state=active]:border-b-2"
            style={{
              color: '#C9D1D9',
              borderColor: '#3B82F6',
            }}
          >
            <DollarSign size={14} />
            Trades
          </Tabs.Trigger>
          <Tabs.Trigger
            value="metrics"
            className="px-4 py-3 text-sm flex items-center gap-2 transition-colors data-[state=active]:border-b-2"
            style={{
              color: '#C9D1D9',
              borderColor: '#3B82F6',
            }}
          >
            <BarChart3 size={14} />
            Metrics
          </Tabs.Trigger>
        </Tabs.List>

        {/* Logs Tab */}
        <Tabs.Content value="logs" className="flex-1 overflow-auto p-4">
          <div className="space-y-1 font-mono text-xs">
            {logs.map((log, index) => (
              <div key={index} className="flex gap-3">
                <span style={{ color: '#9CA3AF' }}>{log.timestamp}</span>
                <span style={{ color: getLevelColor(log.level) }}>
                  [{log.level.toUpperCase()}]
                </span>
                <span style={{ color: '#C9D1D9' }}>{log.message}</span>
              </div>
            ))}
          </div>
        </Tabs.Content>

        {/* Trades Tab */}
        <Tabs.Content value="trades" className="flex-1 overflow-auto p-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{ borderColor: '#1F2933', color: '#9CA3AF' }}>
                <th className="text-left py-2">Time</th>
                <th className="text-left py-2">Pair</th>
                <th className="text-left py-2">Type</th>
                <th className="text-right py-2">Entry</th>
                <th className="text-right py-2">Exit</th>
                <th className="text-right py-2">PnL</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <tr
                  key={trade.id}
                  className="border-b"
                  style={{ borderColor: '#1F2933', color: '#C9D1D9' }}
                >
                  <td className="py-2">{trade.timestamp}</td>
                  <td className="py-2">{trade.pair}</td>
                  <td
                    className="py-2"
                    style={{ color: trade.type === 'BUY' ? '#22C55E' : '#EF4444' }}
                  >
                    {trade.type}
                  </td>
                  <td className="py-2 text-right">${trade.entryPrice.toFixed(2)}</td>
                  <td className="py-2 text-right">
                    {trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : '-'}
                  </td>
                  <td
                    className="py-2 text-right"
                    style={{
                      color: trade.pnl
                        ? trade.pnl >= 0
                          ? '#22C55E'
                          : '#EF4444'
                        : '#9CA3AF',
                    }}
                  >
                    {trade.pnl ? `${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}%` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Tabs.Content>

        {/* Metrics Tab */}
        <Tabs.Content value="metrics" className="flex-1 p-4">
          <div className="grid grid-cols-4 gap-4">
            <div className="p-4 rounded-lg" style={{ backgroundColor: '#1F2933' }}>
              <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
                Total PnL
              </div>
              <div
                className="text-xl"
                style={{
                  color: metrics.totalPnL >= 0 ? '#22C55E' : '#EF4444',
                }}
              >
                {metrics.totalPnL >= 0 ? '+' : ''}
                {metrics.totalPnL.toFixed(2)}%
              </div>
            </div>

            <div className="p-4 rounded-lg" style={{ backgroundColor: '#1F2933' }}>
              <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
                Win Rate
              </div>
              <div className="text-xl" style={{ color: '#C9D1D9' }}>
                {metrics.winRate.toFixed(1)}%
              </div>
            </div>

            <div className="p-4 rounded-lg" style={{ backgroundColor: '#1F2933' }}>
              <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
                Total Trades
              </div>
              <div className="text-xl" style={{ color: '#C9D1D9' }}>
                {metrics.totalTrades}
              </div>
            </div>

            <div className="p-4 rounded-lg" style={{ backgroundColor: '#1F2933' }}>
              <div className="text-xs mb-2" style={{ color: '#9CA3AF' }}>
                Max Drawdown
              </div>
              <div className="text-xl" style={{ color: '#EF4444' }}>
                -{metrics.maxDrawdown.toFixed(2)}%
              </div>
            </div>
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}
