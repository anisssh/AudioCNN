import { useState } from 'react'
import AudioClassifier from './Audioclassifier'



function App() {
  const [count, setCount] = useState(0)

  return (
   <>
   <AudioClassifier/>
   </>
  )
}

export default App
