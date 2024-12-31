import type { Ref } from "vue";
import * as ort from 'onnxruntime-web';

export class AI {
  private session: Ref<ort.InferenceSession>

  constructor() {
    
  }

  act(board: Ref<number[][]>, player1 = 1, player2 = -1): {row: number, col: number} {
    const state = this.preprocess(board, player1, player2);
    const inputTensor = new ort.Tensor("float32", state)
    const ouput = this.session.value.run({ input: inputTensor })
    return {row: -1,  col: -1};
  }

  preprocess(board: Ref<number[][]>, player1: number, player2: number): Array<number> {
    const flattenBoard = board.value.flat();
    const stateSize = flattenBoard.length * 2;

    const statePlayer1 = Array<number>(stateSize / 2).fill(0);
    statePlayer1.map((val, idx) => {
      statePlayer1[idx] = val == player1 ? 1 : 0;
    })
    const statePlayer2 = Array<number>(stateSize / 2).fill(0);
    statePlayer2.map((val, idx) => {
      statePlayer2[idx] = val == player2 ? 1 : 0;
    })

    return statePlayer1.concat(statePlayer2)
  }

}
