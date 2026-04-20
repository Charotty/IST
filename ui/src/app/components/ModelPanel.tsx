import { Brain, Zap, Network } from 'lucide-react';

interface Model {
  name: string;
  type: 'GRU' | 'Transformer' | 'CNN';
  score: number;
  isActive: boolean;
}

interface ModelPanelProps {
  models: Model[];
}

const modelIcons = {
  GRU: Brain,
  Transformer: Network,
  CNN: Zap,
};

export function ModelPanel({ models }: ModelPanelProps) {
  return (
    <div className="p-6 border-t" style={{ borderColor: '#1F2933' }}>
      <div className="text-xs mb-4" style={{ color: '#9CA3AF' }}>
        Model Performance
      </div>
      <div className="space-y-3">
        {models.map((model) => {
          const Icon = modelIcons[model.type];
          return (
            <div
              key={model.name}
              className="p-3 rounded-lg transition-all"
              style={{
                backgroundColor: model.isActive ? '#3B82F620' : '#1F2933',
                border: model.isActive ? '1px solid #3B82F6' : '1px solid transparent',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon
                    size={16}
                    style={{ color: model.isActive ? '#3B82F6' : '#9CA3AF' }}
                  />
                  <span
                    className="text-sm"
                    style={{
                      color: model.isActive ? '#3B82F6' : '#C9D1D9',
                    }}
                  >
                    {model.name}
                  </span>
                </div>
                {model.isActive && (
                  <div
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      backgroundColor: '#3B82F6',
                      color: '#FFFFFF',
                    }}
                  >
                    ACTIVE
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: '#9CA3AF' }}>
                  Score
                </span>
                <span
                  className="text-sm"
                  style={{
                    color: model.score > 0.7 ? '#22C55E' : model.score > 0.5 ? '#F59E0B' : '#EF4444',
                  }}
                >
                  {(model.score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
