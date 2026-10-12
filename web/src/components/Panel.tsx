import { Component, type ErrorInfo, type ReactNode } from "react";
import { RotateCcw, TriangleAlert } from "lucide-react";

interface Props { name: string; children: ReactNode }
interface State { error: Error | null }

/**
 * One panel failing must not take the console down with it. An operator
 * watching a machine keeps the state, the timeline and the other panels; the
 * broken one says so and offers a retry.
 */
export default class Panel extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[panel:${this.props.name}]`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <section className="card">
        <div className="card-b">
          <div className="empty">
            <TriangleAlert size={26} />
            <div><b>{this.props.name}</b> hit an error and was reset.</div>
            <button className="btn sm" onClick={() => this.setState({ error: null })}>
              <RotateCcw size={13} />Show it again</button>
          </div>
        </div>
      </section>
    );
  }
}
