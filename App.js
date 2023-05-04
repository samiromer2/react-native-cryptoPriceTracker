import { StyleSheet, Text, View ,SafeAreaView } from 'react-native';
import Listitem from './components/Listitem';
import { SAMPLE_DATA} from './assets/data/sampleData';
import { FlatList } from 'react-native';

const ListHeader = () => 
(
  <>
   <View style={styles.titleWrapper}>
      <Text style={styles.bigTitle}>Markets</Text>
      </View>
      <View style={styles.divider}/></>
)
export default function App() {
  return (
    <SafeAreaView style={styles.container}>

      <FlatList
      keyExtractor={(item) => item.id}
      data={SAMPLE_DATA}
      renderItem={({item})=> (
        <Listitem
        name = {item.name}
      symbol = {item.symbol}
      currentPrice = {item.current_price}
      priceChangePercentage7d = {item.price_change_percentage_7d_in_currency}
      logoUrl = {item.image}
        />
      )}
      ListHeaderComponent={<ListHeader/>}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  titleWrapper:{
    marginTop:20,
    paddingHorizontal:16,
  },
  bigTitle:{
    fontSize:32,
    fontWeight:'bold',
  },
  divider:{
    height:StyleSheet.hairlineWidth,
    backgroundColor:'#A9ABB1',
    marginHorizontal:16,
    marginTop:16,
  },
});
