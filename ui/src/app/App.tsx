import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { MainChart } from './components/MainChart';
import { SignalPanel } from './components/SignalPanel';
import { ModelPanel } from './components/ModelPanel';
import { BottomPanel } from './components/BottomPanel';

export default function App() {
  const [selectedPair, setSelectedPair] = useState('BTC/USDT');
  const [systemStatus, setSystemStatus] = useState<'RUNNING' | 'STOPPED' | 'ERROR'>('STOPPED');
  const [signal, setSignal] = useState<'BUY' | 'SELL' | 'HOLD'>('HOLD');
  const [confidence, setConfidence] = useState(0.65);
  const [predictedChange, setPredictedChange] = useState(0.0);
  const [logs, setLogs] = useState([
    { timestamp: '14:23:45', level: 'info' as const, message: 'System initialized' },
    { timestamp: '14:23:46', level: 'info' as const, message: 'Connected to Binance API' },
    { timestamp: '14:23:47', level: 'info' as const, message: 'Models loaded successfully' },
  ]);

  const [models] = useState([
    { name: 'GRU-LSTM', type: 'GRU' as const, score: 0.82, isActive: true },
    { name: 'Transformer', type: 'Transformer' as const, score: 0.78, isActive: false },
    { name: 'CNN-Attention', type: 'CNN' as const, score: 0.74, isActive: false },
  ]);

  const [trades] = useState([
    {
      id: '1',
      timestamp: '14:15:32',
      pair: 'BTC/USDT',
      type: 'BUY' as const,
      entryPrice: 45234.50,
      exitPrice: 45567.20,
      pnl: 0.73,
    },
    {
      id: '2',
      timestamp: '14:18:12',
      pair: 'ETH/USDT',
      type: 'SELL' as const,
      entryPrice: 2534.80,
      exitPrice: 2498.30,
      pnl: -1.44,
    },
    {
      id: '3',
      timestamp: '14:22:05',
      pair: 'BTC/USDT',
      type: 'BUY' as const,
      entryPrice: 45456.00,
      exitPrice: undefined,
      pnl: undefined,
    },
  ]);

  const [metrics] = useState({
    totalPnL: 2.34,
    winRate: 67.5,
    totalTrades: 156,
    maxDrawdown: 4.23,
  });

  useEffect(() => {
    if (systemStatus === 'RUNNING') {
      const interval = setInterval(() => {
        const signals: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
        const randomSignal = signals[Math.floor(Math.random() * signals.length)];
        const randomConfidence = 0.4 + Math.random() * 0.5;
        const randomChange = (Math.random() - 0.5) * 2;

        setSignal(randomSignal);
        setConfidence(randomConfidence);
        setPredictedChange(randomChange);

        const now = new Date();
        const timestamp = now.toLocaleTimeString('ru-RU', { hour12: false });

        setLogs((prev) => [
          ...prev.slice(-20),
          {
            timestamp,
            level: 'info' as const,
            message: `Signal: ${randomSignal}, Confidence: ${(randomConfidence * 100).toFixed(1)}%`,
          },
        ]);
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [systemStatus]);

  const handleStart = () => {
    setSystemStatus('RUNNING');
    const now = new Date();
    const timestamp = now.toLocaleTimeString('ru-RU', { hour12: false });
    setLogs((prev) => [
      ...prev,
      { timestamp, level: 'info' as const, message: 'Trading system started' },
    ]);
  };

  const handleStop = () => {
    setSystemStatus('STOPPED');
    const now = new Date();
    const timestamp = now.toLocaleTimeString('ru-RU', { hour12: false });
    setLogs((prev) => [
      ...prev,
      { timestamp, level: 'info' as const, message: 'Trading system stopped' },
    ]);
  };

  return (
    <div className="size-full flex flex-col" style={{ backgroundColor: '#0D1117' }}>
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          onPairSelect={setSelectedPair}
          selectedPair={selectedPair}
          systemStatus={systemStatus}
          onStart={handleStart}
          onStop={handleStop}
        />

        <div className="flex-1 flex flex-col">
          <div className="flex-1 flex overflow-hidden">
            <div className="flex-1">
              <MainChart pair={selectedPair} />
            </div>

            <div className="flex flex-col">
              <SignalPanel
                signal={signal}
                confidence={confidence}
                predictedChange={predictedChange}
                currentModel={models.find((m) => m.isActive)?.name || 'GRU-LSTM'}
              />
              <ModelPanel models={models} />
            </div>
          </div>

          <div style={{ height: '220px' }}>
            <BottomPanel logs={logs} trades={trades} metrics={metrics} />
          </div>
        </div>
      </div>
    </div>
  );
}